import time
import torch
import torchvision
import numpy as np
import torch.distributed as dist

def WarmupLrScheduler( optimizer, warmupIters, warmupFactor ):
# {{{
    def f( x ):

        # 根据step数返回一个学习率倍率因子
        if x >= warmupIters:

            return 1

        alpha = float( x ) / warmupIters

        return warmupFactor * ( 1 - alpha ) + alpha

    return torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda = f )# }}}

def IsDistAvailAndInitialized():
# {{{
    # 检查是否支持分布式环境
    if not dist.is_available():

        return False

    if not dist.is_initialized():

        return False

    return True# }}}

def GetWorldSize():
# {{{
    if not IsDistAvailAndInitialized():

        return 1

    return dist.get_world_size()# }}}

def ReduceDict( inputDict, average = True ):
# {{{
    worldSize = GetWorldSize()

    # 单GPU情况
    if worldSize < 2:

        return inputDict

    # 多GPU情况
    with torch.no_grad():

        names = []
        values = []

        # 对键进行排序，以便它们在各个流程中保持一致
        for k in sorted( inputDict.keys() ):

            names.append( k )
            values.append( inputDict[ k ] )

        values = torch.stack( values, dim = 0 )
        dist.all_reduce( values )
        if average:

            values /= worldSize

        reduceDict = {
                         k : v
                         for k, v in zip( names, values )
                     }

        return reduceDict# }}}

def GetIouTypes( model ):
# {{{
    modelWithoutDdp = model
    if isinstance( model, torch.nn.parallel.DistributedDataParallel ):

        modelWithoutDdp = model.module

    iouTypes = [ 'bbox' ]

    return iouTypes# }}}

def BoxIou( box1, box2 ):
# {{{
    def BoxArea( box ):

        return ( box[ 2 ] - box[ 0 ] ) * ( box[ 3 ] - box[ 1 ] )

    area1 = BoxArea( box1.t() )
    area2 = BoxArea( box2.t() )

    inter = ( torch.min( box1[ :, None, 2 : ], box2[ :, 2 : ] ) - torch.max( box1[ :, None, : 2 ], box2[ :, : 2 ] ) ).clamp( 0 ).prod( 2 )

    return inter / ( area1[ :, None ] + area2 - inter )# }}}

def Nms( pred, confThres = 0.1, iouThres = 0.6, mutiLabel = True, classes = None, agnostic = False, maxNum = 100 ):
# {{{
    '''
        非极大值抑制
        param  : prediction : [ batch, numAnchors * gridH * gridW, numClasses + 1 + 4 ]
        return : nx6 : [ xMin, yMin, xMax, yMax, conf, cls ]
    '''

    merge = False
    minWh, maxWh = 2, 4096
    timeLimit = 10.0

    t = time.time()
    numClasses = pred[ 0 ].shape[ 1 ] - 5
    mutiLabel &= ( numClasses > 1 )
    output = [ None ] * pred.shape[ 0 ]

    # 开始使用约束条件
    for xi, x in enumerate( pred ):

        # 根据objConfidence滤除背景目标
        x = x[ x[ :, 4 ] > confThres ]

        # 滤除小目标
        x = x[ ( ( x[ :, 2 : 4 ] > minWh ) & ( x[ :, 2 : 4 ] < maxWh ) ).all( 1 ) ]

        # 如果没有目标了则进行下一轮
        if not x.shape[ 0 ]:

            continue

        # 计算每个类别的概率conf = objConf * clsConf
        x[ ..., 5 : ] *= x[ ..., 4 : 5 ]

        # 进行坐标转换 x[ centerX, centerY ,width, height ] ---- box[ xMin, yMin, xMax, Ymax ]
        box = torch.zeros_like( x[ :, : 4 ] ) if isinstance( x, torch.Tensor ) else np.zeros_like( x[ :, : 4 ] )
        box[ :, 0 ] = x[ :, 0 ] - x[ :, 2 ] / 2
        box[ :, 1 ] = x[ :, 1 ] - x[ :, 3 ] / 2
        box[ :, 2 ] = x[ :, 0 ] + x[ :, 2 ] / 2
        box[ :, 3 ] = x[ :, 1 ] + x[ :, 3 ] / 2

        # 针对每个类别进行非极大值抑制
        if mutiLabel:

            i, j = ( x[ :, 5 : ] > confThres ).nonzero( as_tuple = False ).t()
            x = torch.cat( ( box[ i ], x[ i, j + 5 ].unsqueeze( 1 ), j.float().unsqueeze( 1 ) ), 1 )

        # 直接针对每个类别中概率最大的类别进行非极大值抑制
        else:

            conf, j = x[ :, 5 : ].max( 1 )
            x = torch.cat( ( box, conf.unsqueeze( 1 ), j.float().unsqueeze( 1 ) ), 1 )[ conf > confThres ]

        # 按类别筛选
        if classes:

            x = x[ ( j.view( -1, 1 ) == torch.tensor( classes, device = j.device ) ).any( 1 ) ]

        # 如果没有剩余图像就进行下一个图像
        n = x.shape[ 0 ]

        if not n:

            continue

        c = x[ :, 5 ] * 0 if agnostic else x[ :, 5 ]
        boxes, scores = x[ :, : 4 ].clone() + c.view( -1, 1 ) * maxWh, x[ :, 4 ]
        i = torchvision.ops.nms( boxes, scores, iouThres )

        # 最多只保留maxNum个目标信息
        i = i[ : maxNum ]

        if merge and ( 1 < n < 3E3 ):

            try:

                iou = BoxIou( boxes[ i ], boxes ) > iouThres
                weights = iou * scores[ None ]

                # 合并boxes
                x[ i, : 4 ] = torch.mm( weights, x[ :, : 4 ] ).float() / weights.sum( 1, keepdim = True )

            except:

                print( x, i, x.shape, i.shape )

        output[ xi ] = x[ i ]

        if ( time.time() - t ) > timeLimit:

            break

    return output# }}}

def ScaleCoords( img1Shape, coords, img0Shape, ratioPad = None ):
# {{{
    '''
        将预测的坐标信息转换回原图像尺度
        img1Shape : 缩放后的图像尺度
        coords    : 预测box信息
        img0Shape : 缩放前的图像尺度
        ratioPad  : 缩放过程中的缩放比例以及pad
    '''

    if ratioPad is None:

        gain = max( img1Shape ) / max( img0Shape )
        pad = ( img1Shape[ 1 ] - img0Shape[ 1 ] * gain ) / 2, ( img1Shape[ 0 ] - img0Shape[ 0 ] * gain ) / 2

    else:

        gain = ratioPad[ 0 ][ 0 ]
        pad = ratioPad[ 1 ]

    coords[ :, [ 0, 2 ] ] -= pad[ 0 ]
    coords[ :, [ 1, 3 ] ] -= pad[ 1 ]
    coords[ :, : 4 ] /= gain
    coords[ :, 0 ].clamp_( 0, img0Shape[ 1 ] )
    coords[ :, 1 ].clamp_( 0, img0Shape[ 0 ] )
    coords[ :, 2 ].clamp_( 0, img0Shape[ 1 ] )
    coords[ :, 3 ].clamp_( 0, img0Shape[ 0 ] )

    return coords# }}}
