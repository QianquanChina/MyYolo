import math
import torch
import torch.nn as nn

def whIou( wh1, wh2 ):
# {{{
    wh1 = wh1[ :, None ]
    wh2 = wh2[ None ]
    inter = torch.min( wh1, wh2 ).prod( 2 )

    return inter / ( wh1.prod( 2 ) + wh2.prod( 2 ) - inter )# }}}

def BboxIou( box1, box2, x1Y1X2Y2 = True, gIou = False, dIou = False, cIou = False ):
# {{{
    box2 = box2.t()

    if x1Y1X2Y2:

        b1X1, b1Y1, b1X2, b1Y2 = box1[ 0 ], box1[ 1 ], box1[ 2 ], box1[ 3 ]
        b2X1, b2Y1, b2X2, b2Y2 = box2[ 0 ], box2[ 1 ], box2[ 2 ], box2[ 3 ]

    else:

        b1X1, b1X2 = box1[ 0 ] - box1[ 2 ] / 2, box1[ 0 ] + box1[ 2 ] / 2
        b1Y1, b1Y2 = box1[ 1 ] - box1[ 3 ] / 2, box1[ 1 ] + box1[ 3 ] / 2
        b2X1, b2X2 = box2[ 0 ] - box2[ 2 ] / 2, box2[ 0 ] + box2[ 2 ] / 2
        b2Y1, b2Y2 = box2[ 1 ] - box2[ 3 ] / 2, box2[ 1 ] + box2[ 3 ] / 2

    inter = ( torch.min( b1X2, b2X2 ) - torch.max( b1X1, b2X1 ) ).clamp( 0 ) * \
            ( torch.min( b1Y2, b2Y2 ) - torch.max( b1Y1, b2Y1 ) ).clamp( 0 )

    w1, h1 = b1X2 - b1X1, b1Y2 - b1Y1
    w2, h2 = b2X2 - b2X1, b2Y2 - b2Y1
    union = ( w1 * h1 + 1e-16 ) + w2 * h2 - inter

    iou = inter / union  # iou

    if gIou or dIou or cIou:

        cw = torch.max(b1X2, b2X2) - torch.min(b1X1, b2X1)
        ch = torch.max(b1Y2, b2Y2) - torch.min(b1Y1, b2Y1)

        if gIou:

            cArea = cw * ch + 1e-16  # convex area

            return iou - (cArea - union) / cArea

        if dIou or cIou:

            c2 = cw ** 2 + ch ** 2 + 1e-16
            rHo2 = ( ( b2X1 + b2X2 ) - ( b1X1 + b1X2 ) ) ** 2 / 4 + ( ( b2Y1 + b2Y2 ) - ( b1Y1 + b1Y2 ) ) ** 2 / 4

            if dIou:

                return iou - rHo2 / c2  # DIoU

            elif cIou:

                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

                with torch.no_grad():

                    alpha = v / (1 - iou + v)

                return iou - (rHo2 / c2 + v * alpha)

    return iou# }}}

class Yolo_Loss( nn.Module ):

    def __init__( self, device, model ):
# {{{
        super().__init__()
        self.device = device
        self.model = model# }}}

    def BuildTarget( self, pred, targets ):
# {{{
        # targets : [ imageIdx, class, x, y, w, h ]
        # imageIdx : 此目标属于当前batch中的哪个图片的目标
        # x, y, w, h : 相对位置
        numTargets = targets.shape[ 0 ]
        targetCls, targetBox, indices, anchors = [], [], [], []
        gain = torch.ones( 6, device = targets.device ).long()
        multiGpu = type( self.model ) in ( nn.parallel.DataParallel, nn.parallel.DistributedDataParallel )

        for i, j in enumerate( self.model.yoloLayers ):

            # 获取该yolo predictor对应的anchors, anchorVec是anchor缩放到对应特征层的尺度
            anchor = self.model.module.moduleList[ j ].anchorVec if multiGpu else self.model.moduleList[ j ].anchorVec

            # pred : [ batchSize, numAnchors, gridH, gridW, numParams ]
            gain[ 2 : ] = torch.tensor( pred[ i ].shape )[ [ 3, 2, 3, 2 ] ]

            numAnchors = anchor.shape[ 0 ]
            anchorsTensor = torch.arange( numAnchors ).view( numAnchors, 1 ).repeat( 1, numTargets )

            a, t, offset = [], targets * gain, 0

            if numTargets:

                j = whIou( anchor, t[ :, 4 : 6 ] ) > self.model.hyp[ 'iou_t' ]
                a, t = anchorsTensor[ j ], t.repeat( numAnchors, 1, 1 )[ j ]

            b, c = t[ :, : 2].long().T
            gXy = t[ :, 2 : 4 ]
            gWh = t[ :, 4 : 6 ]
            gIj = ( gXy - offset ).long()
            gi, gj = gIj.T
            indices.append( ( b, a, gj.clamp_( 0, gain[ 3 ] - 1 ), gi.clamp_( 0, gain[ 2 ] - 1 ) ) )
            targetBox.append( torch.cat( ( gXy - gIj, gWh), 1 ) )
            anchors.append( anchor[ a ] )
            targetCls.append( c )

            if c.shape[ 0 ]:

                assert c.max() < self.model.nc, 'model accepts %g classes labeled from 0-%g, however you labelled to class %g' \
                                                % ( self.model.nc, self.model.nc - 1, c.max() )

        return targetCls, targetBox, indices, anchors# }}}

    def forward( self, pred, targets ):
# {{{
        lossCls = torch.zeros( 1, device = self.device )
        lossBox = torch.zeros( 1, device = self.device )
        lossObj = torch.zeros( 1, device = self.device )
        targetCls, targetBox, indices, anchors = self.BuildTarget( pred, targets )
        bceCls = nn.BCEWithLogitsLoss( pos_weight = torch.tensor( [ self.model.hyp[ 'cls_pw' ] ], device = self.device  ), reduction = 'mean' )
        bceObj = nn.BCEWithLogitsLoss( pos_weight = torch.tensor( [ self.model.hyp[ 'obj_pw' ] ], device = self.device  ), reduction = 'mean' )
        clsPositive, clsNegative = 1, 0

        for i, pi in enumerate( pred ):

            # imageIdx, anchorIdx, girdY, gridX
            b, a, gj, gi = indices[ i ]
            targetObj = torch.zeros_like( pi[ ..., 0 ], device = self.device )

            # 正样本的个数
            nb = b.shape[ 0 ]

            if nb:

                # 对应匹配到正样本的预测信息
                ps = pi[ b, a, gj, gi ]
                # GIOU
                pXy = ps[ :, : 2 ].sigmoid()
                pWh = ps[ :, 2 : 4 ].exp().clamp( max = 1E3 ) * anchors[ i ]
                pBox = torch.cat( ( pXy, pWh ), 1 )
                gIou = BboxIou( pBox.t(), targetBox[ i ], x1Y1X2Y2 = False, gIou = True )
                lossBox += ( 1.0 - gIou ).mean()
                targetObj[ b, a, gj, gi ] = ( 1.0 - self.model.gr ) + self.model.gr * gIou.detach().clamp( 0 ).type( targetObj.dtype )

                if self.model.nc > 1:

                    t = torch.full_like( ps[ :, 5 : ], clsNegative, device = self.device )
                    t[ range( nb ), targetCls[ i ] ] = clsPositive
                    lossCls += bceCls( ps[ :, 5 : ], t )

            lossObj += bceObj( pi[ ..., 4 ], targetObj )

        lossBox *= self.model.hyp[ 'giou' ]
        lossObj *= self.model.hyp[ 'obj'  ]
        lossCls *= self.model.hyp[ 'cls'  ]

        return { 'boxLoss' : lossBox, 'objLoss' : lossObj, 'clsLoss' : lossCls }# }}}

