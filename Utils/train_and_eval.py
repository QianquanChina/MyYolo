import sys
import time
import math
import torch
import random
import torch.nn.functional as F
from torch.cuda import amp
from .metric_logger import Metric_Logger, Smoothed_Value
from .coco_tool import GetCocApiFromDataSet, Coc_Evaluator
from .build_tool import WarmupLrScheduler, ReduceDict, GetIouTypes, Nms, ScaleCoords

def TrainOneEpoch(
                     model         , lossFunc      ,optimizer,
                     trainDataLoder, device        ,epoch    ,
                     printFreq     , accumulate    ,imgSize  ,
                     gridMin       , gridMax       ,gridSize ,
                     multiScale = False,
                     warmUp     = False,
                     scaler     = None
                 ):
# {{{
    model.train()
    merticLogger = Metric_Logger( delimiter = ' ' )
    merticLogger.add_meter( 'lr', Smoothed_Value( window_size = 1, fmt = '{value:.6f}' ) )
    header = 'Epoch : [ {} ]'.format( epoch )

    lrScheduler = None

    if epoch == 0 and warmUp is True:

        warmupFactor = 1.0 / 1000
        warmupIters = min( 1000, len( trainDataLoder ) - 1 )
        lrScheduler = WarmupLrScheduler( optimizer, warmupIters, warmupFactor )
        accumulate = 1

    meanLoss = torch.zeros( 4 ).to( device )
    nowLr = 0.
    numBatch = len( trainDataLoder )
    model.to( device )

    for i, ( imgs, targets, paths, _, _, ) in enumerate( merticLogger.log_every( trainDataLoder, printFreq, header ) ):

        # numAllBatch 统计从epoch0 开始的所有的batch数
        numAllBatch = i + numBatch * epoch
        imgs, targets = imgs.to( device ).float() / 255.0, targets.to( device )

        # 多尺度
        if multiScale:

            # 每训练64张图片，就随机修改一次输入图片的大小，由于label已变成相对坐标，故缩放图片不影响label的值
            if numAllBatch % accumulate == 0:

                imgSize = random.randrange( gridMin, gridMax + 1 ) * gridSize

            scaleFactor = imgSize / max( imgs.shape[ 2 : ] )

            # 如果图片最大边长不等于imgSize，则缩放图片，并将长和宽调整到32的整数倍
            if scaleFactor != 1:

                newShape = [
                               math.ceil( x * scaleFactor / gridSize ) * gridSize
                               for x in imgs.shape[ 2 : ]
                           ]
                imgs = F.interpolate( imgs, size = newShape, mode = 'bilinear', align_corners = False )

        # 混合精度训练上下文管理器，在CPU环境中无任何作用
        with amp.autocast_mode.autocast( enabled = scaler is not None ):

            pred = model( imgs )
            lossDict = lossFunc( pred, targets )
            losses = sum(
                            loss
                            for loss in lossDict.values()
                        )

        # 减少所有GPU上的损耗，以便记录
        lossDictReduced = ReduceDict( lossDict )
        lossesReduced = sum(
                               loss
                               for loss in lossDictReduced.values()
                           )

        lossItems = torch.cat(
                                 (
                                    lossDictReduced[ 'boxLoss' ],
                                    lossDictReduced[ 'objLoss' ],
                                    lossDictReduced[ 'clsLoss' ],
                                    lossesReduced #type:ignore
                                 )
                             ).detach()

        meanLoss = ( meanLoss * i + lossItems ) / ( i + 1 )

        if not torch.isfinite( lossesReduced ): #type:ignore

            print( 'Warning : non-finite loss, ending training', lossesReduced )
            print( 'training image path : {}'.format( ','.join( paths) ) )
            sys.exit( 1 )

        losses *= 1. / accumulate

        # backward
        if scaler is not None:

            scaler.scale( losses ).backward()

        else:

            losses.backward() #type:ignore

        # optimize
        # 每训练64张图片更新一次权重
        # accumulate : 迭代多少个batch才能训练完64张图片
        if numAllBatch % accumulate == 0:

            if scaler is not None:

                scaler.step( optimizer )
                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad()

        nowLr = optimizer.param_groups[ 0 ][ 'lr' ]
        merticLogger.update( loss = lossesReduced, lr = nowLr, **lossDictReduced )

        # 第一轮使用warmup训练方式
        if numAllBatch % accumulate == 0 and lrScheduler is not None:

            lrScheduler.step()

    return meanLoss, nowLr# }}}

@torch.no_grad()
def Evaluate( model, valDataLoder, printFreq, coco = None, device = None ):
# {{{
    cupDevice = torch.device( 'cpu' )
    model.eval()
    metricLog = Metric_Logger( delimiter = ' ' )
    header = 'Valid :'

    if coco is None:

        coco = GetCocApiFromDataSet( valDataLoder.dataset )

    iouTypes = GetIouTypes( model )
    cocoEvaluator = Coc_Evaluator( coco, iouTypes )
    model = model.to( device )

    for imgs, targets, _, shapes, imgIndex in metricLog.log_every( valDataLoder, printFreq, header ):

        imgs, targets = imgs.to( device ).float() / 255.0, targets.to( device )

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device( 'cpu' ):

            torch.cuda.synchronize( device )

        modelTime = time.time()
        pred = model( imgs )[ 0 ]
        pred = Nms( pred, confThres = 0.01, iouThres = 0.6, mutiLabel = False )
        modelTime = time.time() - modelTime

        outputs = []

        for index, p in enumerate( pred ):

            if p is None:

                p = torch.empty( ( 0, 6 ), device = cupDevice )
                boxes = torch.empty( ( 0, 4 ), device = cupDevice )

            else:

                boxes = p[ :, : 4 ]
                # 将boxes信息还回原来尺度，这样计算的mAP才是准确的
                boxes = ScaleCoords( imgs[ index ].shape[ 1 : ], boxes, shapes[ index ][ 0 ] ).round()

            # 注意这里传入的boxes格式必须是xMin, yMin, xMax, yMax,且是绝对坐标
            info = {
                       'boxes' : boxes.to( cupDevice ),
                       'labels' : p[ :, 5 ].to( device = cupDevice, dtype = torch.int64 ),
                       'scores' : p[ :, 4 ].to( cupDevice )
                   }
            outputs.append( info )

        res = {
                  imgId : output
                  for imgId, output in zip( imgIndex, outputs )
              }

        evaluateTime = time.time()
        cocoEvaluator.Update( res )
        evaluateTime = time.time() - evaluateTime
        metricLog.update( modelTime = modelTime, evaluateTime = evaluateTime )

    print( 'Averaged stats : ', metricLog )
    cocoEvaluator.SynchronizeBetweenProcesses()
    cocoEvaluator.Accumulate()
    cocoEvaluator.Summarize()
    resultInfo = cocoEvaluator.cocoEval[ iouTypes[ 0 ] ].stats.tolist()

    return resultInfo# }}}
