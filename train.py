import os
import yaml
import glob
import math
import torch
import datetime
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from Backbone import Dark_Net, Yolo_Layer
from torch.utils.tensorboard.writer import SummaryWriter
from Utils import ParseDataCfg, Data_Set, GetCocApiFromDataSet, TrainOneEpoch, Evaluate, Yolo_Loss

def main( hyp, opt ):
# {{{
    device = torch.device( opt.device if torch.cuda.is_available() else 'cpu' )
    print( 'Using {} device training'.format( device.type ) )

    weightsDir = 'Weights' + os.sep

    if os.path.exists( weightsDir ) is False:

        os.makedirs( weightsDir )

    best = weightsDir + 'best.pt'
    resultsFile = 'results{}.txt'.format( datetime.datetime.now().strftime( '%Y%m%d-%H%M%S' ) )

    cfg          = opt.cfg
    data         = opt.data
    epochs       = opt.epochs
    weights      = opt.weights
    batchSize    = opt.batchSize
    imgSizeTest  = opt.imgSize
    multiScale   = opt.multiScale
    imgSizeTrain = opt.imgSize

    # 每训练64张图片才更新一次权重
    accumulate   = max( round( 64 / batchSize ), 1 )

    # 将图片设置成32的倍数
    gridSize = 32
    assert math.fmod( imgSizeTest, gridSize ) == 0, '--imgSize %g must be a %g-multiple' %( imgSizeTest, gridSize )
    gridMin, gridMax = imgSizeTest // gridSize, imgSizeTest // gridSize

    if multiScale:

        imgSizeMin = opt.imgSize // 1.5
        imgSizeMax = opt.imgSize // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        gridMin, gridMax = imgSizeMin // gridSize, imgSizeMax // gridSize
        imgSizeMin, imgSizeMax = int( gridMin * gridSize ), int( gridMax * gridSize )
        imgSizeTrain = imgSizeMax
        print( 'Using multiScale training, image range[ {}, {} ]'.format( imgSizeMin, imgSizeMax ) )

    dataDict = ParseDataCfg( data )
    trainPath = dataDict[ 'train' ]
    valPath = dataDict[ 'val' ]
    nc = 1 if opt.singleCls else int( dataDict[ 'classes' ] )

    # 调整了class loss 和 obj loss 的损失增益
    hyp[ 'cls' ] *= nc / 80
    hyp[ 'obj' ] *= imgSizeTest / 320

    # 移除之前的结果
    for f in glob.glob( resultsFile ):

        os.remove( f )

    model = Dark_Net( cfg )

    # 是否冻结权重，只训练predictor的权重
    if opt.freezeLayers:

        # 索引减一就是predictor的索引，Yolo_Layer不是predictor
        outputLayerIndices = [
                                 index - 1
                                 for index, module in enumerate( model.moduleList )
                                 if isinstance( module, Yolo_Layer )
                             ]

        # 冻结除predictor和Yolo_Layer之外的所有层
        freezeLayerIndices = [

                                 x
                                 for x in range( len( model.moduleList ) )
                                 if ( x not in outputLayerIndices ) and
                                 ( x - 1 not in outputLayerIndices )
                             ]

        for index in freezeLayerIndices:

            for paramter in model.moduleList[ index ].parameters():

                paramter.requires_grad_( False )

    else:

        # 默认只训练Dark_Net之后的部分
        darkNetEndLayer = 74

        for index in range( darkNetEndLayer + 1 ):

            for paramter in model.moduleList[ index ].parameters():

                paramter.requires_grad_( False )

    pg = [
             p
             for p in model.parameters()
             if p.requires_grad
         ]
    optimizer = optim.SGD( pg, lr = hyp[ 'lr0' ], momentum = hyp[ 'momentum' ], weight_decay = hyp[ 'weight_decay' ], nesterov = True )
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if opt.amp else None

    startEpoch = 0
    bestMap = 0.0

    if weights.endswith( '.pt' ) or weights.endswith( '.pth' ):

        ckpt = torch.load( weights, map_location = device )

        # 加载模型
        try:

            ckpt[ 'model' ] = {
                                  k : val
                                  for k, val in ckpt[ 'model' ].items()
                                  if model.state_dict()[ k ].numel() == val.numel()
                              }
            model.load_state_dict( ckpt[ 'model' ], strict = False )

        except KeyError as e:

            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                'See https://github.com/ultralytics/yolov3/issues/657' % ( opt.weights, opt.cfg, opt.weights )
            raise KeyError( s ) from e

        # 加载优化器
        if ckpt[ 'optimizer' ] is not None:

            optimizer.load_state_dict( ckpt[ 'optimizer' ] )

            if 'bestMap' in ckpt.keys():

                bestMap = ckpt[ 'bestMap' ]

        # 加载结果
        if ckpt.get( 'trainingResults' ) is not None:

            with open( resultsFile, 'w' ) as file:

                file.write( ckpt[ 'trainingResults' ] )

        # 加载epoch
        startEpoch = ckpt[ 'epoch' ] + 1

        if epochs < startEpoch:

            print( '%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' % ( opt.weights, ckpt[ 'epoch'], epochs ) )
            epochs += ckpt[ 'epoch' ]

        if opt.amp and 'scaler' in ckpt:

            scaler.load_state_dict( ckpt[ 'scaler' ] ) #type:ignore

        del ckpt

    lf = lambda x : ( ( 1 + math.cos( x * math.pi / epochs ) ) / 2 ) * ( 1 - hyp[ 'lrf' ] ) + hyp[ 'lrf' ]
    scheduler = optim.lr_scheduler.LambdaLR( optimizer, lr_lambda = lf )
    scheduler.last_epoch = startEpoch #type:ignore

    # 训练集的图像尺寸指定为multiScaleRange中的最大尺寸
    trainDataSet = Data_Set(
                               trainPath, imgSizeTrain, batchSize,
                               augment   = True,
                               hyp       = hyp,
                               rect      = opt.rect,
                               singleCls = opt.singleCls
                           )

    # 验证集的图像尺寸指定为imgSize
    valDataSet = Data_Set(
                             valPath, imgSizeTest, batchSize,
                             augment   = False,
                             hyp       = hyp,
                             rect      = True,
                             singleCls = opt.singleCls
                         )
    nw = min(
                [   #type:ignore
                    os.cpu_count(), batchSize
                    if batchSize > 1
                    else 0,
                    8
                ]
            )
    trainDataLoader = DataLoader(
                                    trainDataSet,
                                    batch_size  = batchSize,
                                    num_workers = nw,
                                    shuffle     = not opt.rect,
                                    collate_fn  = trainDataSet.collate_fn
                                )
    valDataLoader = DataLoader(
                                  valDataSet,
                                  batch_size  = batchSize,
                                  num_workers = nw,
                                  collate_fn  = valDataSet.collate_fn
                              )

    model.nc = nc #type:ignore
    model.hyp = hyp
    model.gr = 1.0 #type:ignore
    lossFunc = Yolo_Loss( device, model )
    coco = GetCocApiFromDataSet( valDataSet )
    print( 'starting training for %g epochs...' % epochs )
    print( 'using %g dataLoader workers' % nw )

    for epoch in range( startEpoch, epochs ):

        meanLoss, lr = TrainOneEpoch(
                                        model, lossFunc, optimizer, trainDataLoader, device, epoch,
                                        accumulate = accumulate, imgSize  = imgSizeTrain,
                                        multiScale = multiScale, gridMin  = gridMin     ,
                                        gridMax    = gridMax   , gridSize = gridSize    ,
                                        printFreq  = 50        , warmUp   = True        ,
                                        scaler     = scaler    ,
                                    )

        scheduler.step()

        if opt.notest is False or epoch == epochs - 1:

            resultInfo = Evaluate( model, valDataLoader, printFreq = 100, coco = coco, device = device )
            cocoMap = resultInfo[ 0 ] #type:ignore
            vocMap  = resultInfo[ 1 ] #type:ignore
            cocoMar = resultInfo[ 8 ] #type:ignore

            if tbWriter:

                tags = [
                           'train/giouLoss',
                           'train/objLoss' ,
                           'train/clsLoss' ,
                           'train/loss'    ,
                           'learningRate'  ,
                           'mAP@[ IoU = 0.50 : 0.95 ]',
                           'mAP@[ IoU = 0.5 ]'        ,
                           'mAR@[ IoU = 0.50 : 0.95 ]'
                       ]

                for x, tag in zip( meanLoss.tolist() + [ lr, cocoMap, vocMap, cocoMar ], tags ):

                    tbWriter.add_scalar( tag, x, epoch )

            with open( resultsFile, 'a' ) as f:

                resultInfo = [
                                 str( round( i, 4 ) )
                                 for i in resultInfo + [ meanLoss.tolist()[ -1 ] ] #type:ignore
                             ] + \
                             [
                                 str( round( lr, 6 ) )
                             ]
                txt = 'epoch : {} {}'.format( epoch, ' '.join( resultInfo ) )
                f.write( txt + '\n' )

            if cocoMap > bestMap:

                bestMap = cocoMap

            if opt.saveBest is False:

                with open( resultsFile, 'r' ) as f:

                    saveFiles = {
                                    'model' : model.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                    'trainingResults' : f.read(),
                                    'epoch' : epoch,
                                    'bestMap' : bestMap
                                }

                    if opt.amp:

                        saveFiles[ 'scaler' ] = scaler.state_dict() #type:ignore

                    torch.save( saveFiles, './Weights/yolov3_spp_{}.pt'.format( epoch ) )

            else:

                if bestMap == cocoMap:

                    with open( resultsFile, 'r' ) as f:

                        saveFiles = {
                                        'model' : model.state_dict(),
                                        'optimizer' : optimizer.state_dict(),
                                        'trainingResults' : f.read(),
                                        'epoch' : epoch,
                                        'bestMap' : bestMap
                                    }

                        if opt.amp:

                            saveFiles[ 'scaler' ] = scaler.state_dict() #type:ignore

                        torch.save( saveFiles, best.format( epoch ) )# }}}

if __name__ == '__main__':
# {{{
    parser = argparse.ArgumentParser()
    parser.add_argument( '--epochs'      , type = int , default = 30 )
    parser.add_argument( '--batchSize'   , type = int , default = 2  )
    parser.add_argument( '--imgSize'     , type = int , default = 512                     , help = 'test size'                      )
    parser.add_argument( '--cfg'         , type = str , default = './Config/my_yolov3.cfg', help = '*.cfg path'                     )
    parser.add_argument( '--data'        , type = str , default = './Data/my_data.data'   , help = '*.data path'                    )
    parser.add_argument( '--hyp'         , type = str , default = './Config/hyp.yaml'     , help = '*.yaml path'                    )
    parser.add_argument( '--weights'     , type = str , default = ''                      , help = 'initial weights path'           )
    parser.add_argument( '--multiScale'  , type = bool, default = True                    , help = 'adjust imgSize every 10 batchs' )
    parser.add_argument( '--saveBest'    , type = bool, default = False                   , help = 'only save best checkpoint'      )
    parser.add_argument( '--freezeLayers', type = bool, default = False                   , help = 'Freeze non output layers'       )

    parser.add_argument( '--rect'     , default = False, action = 'store_true', help = 'rectangular training'             )
    parser.add_argument( '--notest'   , default = False, action = 'store_true', help = 'only test final epoch'            )
    parser.add_argument( '--cacheImg' , default = False, action = 'store_true', help = 'cache images for faster training' )
    parser.add_argument( '--singleCls', default = False, action = 'store_true', help = 'train as singleCls dataset'       )

    parser.add_argument( '--name'  , default = ''      , help = 'rename results.txt to results_name.txt if supplied' )
    parser.add_argument( '--device', default = 'cuda:0', help = 'device id ( i.e. 0 or 0,1 or cpu )'                 )
    parser.add_argument( '--amp'   , default = True    , help = 'Ues torch.cuda.amp for mixed precison training'     )

    opt = parser.parse_args()

    assert os.path.exists( opt.cfg  ), '.cfg file not exists'
    assert os.path.exists( opt.hyp  ), '.hyp file not exists'
    assert os.path.exists( opt.data ), '.data file not exists'

    with open( opt.hyp ) as f:

        hyp = yaml.load( f, Loader = yaml.FullLoader )

    print( "Start Tensorboard with 'tensorboard --logdir=runs', view at http://localhost:6006/" )
    tbWriter = SummaryWriter( comment = opt.name )
    torch.set_printoptions( precision = 5 )

    main( hyp, opt )# }}}
