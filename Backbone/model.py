import sys
import math
import torch
import torch.nn as nn

sys.path.append( '..' )
from Utils import ParseModelCfg, Feature_Concat, Weighted_Feature_Fusion

ONNX_EXPORT = False

def GetYoloLayers( self ):
# {{{
    '''
        获取三个yolo layer模块所对应的索引
    '''

    return [ i for i, m in enumerate( self.moduleList ) if m.__class__.__name__ == 'Yolo_Layer' ]# }}}

def CreateModules( moduleDefs, imgSize ):
# {{{
    '''
        通过moduleDefs构建网络结构
    '''

    imgSize = [ imgSize ] * 2 if isinstance( imgSize, int ) else imgSize

    # 因为不会使用到cfg列表中的第一个配置（[net]的配置）所以删除
    moduleDefs.pop( 0 )
    outputFilters = [ 3 ]
    moduleList = nn.ModuleList()

    # routs : 统计某些特征层在网络中的位置，这些位置的网络输出会被后续层使用到（特征融合或者拼接）
    routs = []
    yoloIndex = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate( moduleDefs ):

        modules = nn.Sequential()

        if mdef[ 'type' ] == 'convolutional':

            bn = mdef[ 'batch_normalize' ]
            filters = mdef[ 'filters' ]
            k = mdef[ 'size' ]
            stride = mdef[ 'stride' ] if 'stride' in mdef else ( mdef[ 'stride_y' ], mdef[ 'stride_x' ] )

            if isinstance( k, int ):

                modules.add_module(
                                      'conv2Dw',
                                      nn.Conv2d(
                                                   in_channels  = outputFilters[ -1 ],
                                                   out_channels = outputFilters[ -1 ],
                                                   kernel_size  = k,
                                                   stride       = stride,
                                                   padding      = k // 2 if mdef[ 'pad' ] else 0,
                                                   groups       = outputFilters[ -1 ],
                                                   bias         = not bn
                                               )
                                  )
                modules.add_module(
                                      'conv2Pw',
                                      nn.Conv2d(
                                                   in_channels  = outputFilters[ -1 ],
                                                   out_channels = filters,
                                                   kernel_size  = 1,
                                                   stride       = 1,
                                                   bias         = not bn
                                               )
                                  )
            else:

                raise TypeError( 'conv2d filters size must be in type' )

            if bn:

                modules.add_module( 'BatchNorm2d', nn.BatchNorm2d( filters ) )

            else:

                # 这个网络除了最后的三个predictor之外的所有的convolutional都是有bn层的
                routs.append( i )

            if mdef[ 'activation' ] == 'leaky':

                modules.add_module( 'activation', nn.LeakyReLU( 0.1, inplace = True ) )

            else:

                pass

        elif mdef[ 'type' ] == 'predconvolutional':

            bn = mdef[ 'batch_normalize' ]
            filters = mdef[ 'filters' ]
            k = mdef[ 'size' ]
            stride = mdef[ 'stride' ] if 'stride' in mdef else ( mdef[ 'stride_y' ], mdef[ 'stride_x' ] )

            if isinstance( k, int ):

                modules.add_module(
                                      'Conv2d',
                                      nn.Conv2d(
                                                   in_channels  = outputFilters[ - 1 ],
                                                   out_channels = filters,
                                                   kernel_size  = k,
                                                   stride       = stride,
                                                   padding      = k // 2 if mdef[ 'pad' ] else 0,
                                                   bias         = not bn
                                               )
                                  )
            else:

                raise TypeError( 'conv2d filters size must be in type' )

            if bn:

                modules.add_module( 'BatchNorm2d', nn.BatchNorm2d( filters ) )

            else:

                # 这个网络除了最后的三个predictor之外的所有的convolutional都是有bn层的
                routs.append( i )

            if mdef[ 'activation' ] == 'leaky':

                modules.add_module( 'activation', nn.LeakyReLU( 0.1, inplace = True ) )

            else:

                pass

        elif mdef[ 'type' ] == 'BatchNorm2d':

            pass

        elif mdef[ 'type' ] == 'maxpool':

            k = mdef[ 'size' ]
            stride = mdef[ 'stride' ]
            modules = nn.MaxPool2d( kernel_size = k, stride = stride, padding = ( k - 1 ) // 2 )

        elif mdef[ 'type' ] =='upsample':

            if ONNX_EXPORT:

                g = ( yoloIndex + 1 ) * 2 / 32
                modules = nn.Upsample( size = tuple( int( x * g ) for x in imgSize ) )

            else:

                modules = nn.Upsample( scale_factor = mdef[ 'stride' ] )

        elif mdef[ 'type' ] == 'route':

            layers = mdef[ 'layers' ]
            filters = sum( [ outputFilters[ l + 1 if l > 0 else l ] for l in layers ] )
            routs.extend( [ i + l if l < 0 else l for l in layers ] )
            modules = Feature_Concat( layers = layers )

        elif mdef[ 'type' ] == 'shortcut':

            layers = mdef[ 'from' ]
            filters = outputFilters[ -1 ]
            routs.append( i + layers[ 0 ] )
            modules = Weighted_Feature_Fusion( layers = layers, weight = 'weights_type' in mdef )

        elif mdef[ 'type' ] == 'yolo':

            yoloIndex += 1

            # 预测层对应原图的比例
            stride = [ 32, 16, 8 ]
            modules = Yolo_Layer(
                                    anchors = mdef[ 'anchors' ][ mdef[ 'mask' ] ],
                                    nc      = mdef[ 'classes' ],
                                    imgSize = imgSize,
                                    stride  = stride[ yoloIndex ]
                                )

            try:

                # yolo layers的上一层
                j = -1
                b = moduleList[ j ][ 0 ].bias.view( modules.na, -1 ) #type:ignore
                b.data[ :, 4 ] += -4.5
                b.data[ :, 5 : ] += math.log( 0.6 / ( modules.nc - 0.99 ) )
                moduleList[ j ][ 0 ].bias = torch.nn.parameter.Parameter( b.view( -1 ), requires_grad = True) #type:ignore

            except Exception as e:

                print( 'WARNING: smart bias initialization failure.', e )
        else:

            print( 'Warning: Unrecognized Layer Type: ' + mdef[ 'type' ] )

        moduleList.append( modules )
        outputFilters.append( filters ) #type:ignore

    routsBinary = [ False ] * len( moduleDefs )

    for i in routs:

        routsBinary[ i ] = True

    return moduleList, routsBinary# }}}

class Yolo_Layer( nn.Module ):
# {{{
    def __init__( self, anchors, nc, imgSize, stride ):
# {{{
        super().__init__()

        self.anchores = torch.Tensor( anchors )

        # 特征层相对原图的步距
        self.stride = stride

        # anchors的数量
        self.na = len( anchors )

        # 训练的类别数
        self.nc = nc

        # 每一个anchor对应的输出维度( x, y, w, h, obj, nc )
        self.no = nc + 5

        # nx,ny是预测特征层的的宽度和高度 ng是gridcell的size（此处是进行简单的初始化）
        self.nx, self.ny, self.ng = 0, 0, ( 0, 0 )

        # 将anchors大小缩放到grid尺度
        self.anchorVec = self.anchores / self.stride

        # batchSize, anchorNum, gridH, gridW, anchorWh 因为batchSzie的个数是不一定的，以及网格的高宽随着
        # 我们的输入不同其个数也是不一定的所以这里都给设置成1，后续根据广播机制就可以自动填充
        self.anchorWh = self.anchorVec.view( 1, self.na, 1, 1, 2 )
        self.grid = None

        if ONNX_EXPORT:

                    self.training = False
                    self.CreateGrids( ( imgSize[ 1 ] // stride, imgSize[0] // stride ) )# }}}

    def CreateGrids( self, ng = ( 13, 13 ), device = 'cpu' ):
# {{{
        """
            更新grids信息并生成新的grids参数
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor( ng, dtype = torch.float )

        # 构建每个grid cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:

            # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid(
                                       [
                                           torch.arange( self.ny, device = device ),
                                           torch.arange( self.nx, device = device )
                                       ]
                                   )
            # batchSize, na, gridH, gridW, xy
            self.grid = torch.stack( ( xv, yv ), 2 ).view( ( 1, 1, self.ny, self.nx, 2 ) ).float()

        if self.anchorVec.device != device:

            self.anchorVec = self.anchorVec.to( device )
            self.anchorWh = self.anchorWh.to( device )# }}}

    def forward( self, p ):
# {{{
        if ONNX_EXPORT:

            bs = 1

        else:

            # p.shape = [ batchSize, predictParam( 25 * 3 ), gridH( 16( 32、64 ) ), gridW( 16( 32、64 ) ) ]
            bs, _, ny, nx = p.shape

            if ( self.nx, self.ny ) != ( nx, ny ) or self.grid is None:

                self.CreateGrids( ( nx, ny ), p.device )

        # p.shape = [ batchSize, anchorNum( 3 ), gridH( 16( 32、64 ) ), gridW( 16( 32、64 ) ), eachAnchorPredicParam( 25 ) ]
        p = p.reshape( bs, self.na, self.no, self.ny, self.nx ).permute( 0, 1, 3, 4, 2 )

        if self.training:

            return p

        elif ONNX_EXPORT:

            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat( m, 1 ) #type:ignore
            grid = self.grid.repeat( 1, self.na, 1, 1, 1 ).view( m, 2 ) #type:ignore
            anchorWh = self.anchorWh.repeat( 1, 1, self.nx, self.ny, 1 ).view( m, 2 ) * ng

            p = p.view( m, self.no )
            p[ :, : 2 ] = ( torch.sigmoid( p[ :, 0 : 2 ] ) + grid ) * ng
            p[ :, 2 : 4 ] = torch.exp( p[ :, 2 : 4 ] ) * anchorWh
            p[ :, 4 : ] = torch.sigmoid( p[ :, 4 : ] )
            p[ :, 5 : ] = p[ :, 5 : self.no ] * p[ :, 4 : 5 ]

            return p

        else:

            io = p.clone()
            # 计算在feature map上的xy坐标
            io[ ..., : 2 ] = torch.sigmoid( io[ ..., : 2 ] ) + self.grid
            io[ ..., 2 : 4 ] = torch.exp( io[ ..., 2 : 4 ] ) * self.anchorWh
            io[ ..., : 4 ] *= self.stride
            torch.sigmoid_( io[ ..., 4 : ] )
            return io.reshape( bs, -1, self.no ), p# }}}}}}

class Dark_Net( nn.Module ):
# {{{
    def __init__( self, cfg, imgSize = ( 416, 416 ), verbose = False ):
# {{{
        super().__init__()

        # imgSize只在导出ONNX模型时起作用
        self.inputSize = [ imgSize ] * 2 if isinstance( imgSize, int ) else imgSize

        # 解析网络的对应的.cfg文件
        self.moduleDefs = ParseModelCfg( cfg )

        # 根据解析的参数去构建网络
        self.moduleList, self.routs = CreateModules( self.moduleDefs, imgSize )

        # 获得所有Yolo_Layer层的索引
        self.yoloLayers = GetYoloLayers( self )

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.Info( verbose ) if not ONNX_EXPORT else None# }}}

    def forward( self, x, verbose = False ):
# {{{
        # yoloOut收集每个yoloLayer层的输出
        # out收集每个模块的输出
        yoloOut, out = [], []

        if verbose:

            print( '0', x.shape )
            str = ''

        for i, module in enumerate( self.moduleList ):

            name = module.__class__.__name__

            if name in [ 'Weighted_Feature_Fusion', 'Feature_Concat' ]:

                if verbose:

                    l = torch.Tensor( [ i - 1 ] ) + module.layers
                    sh = [ list( x.shape ) ] + [ list( out[ i ].shape ) for i in module.layers ] #type:ignore
                    str = ' >> ' + ' + '.join( [ 'layer %g %s' % x for x in zip( l, sh ) ] )
                x = module( x, out )

            elif name == 'Yolo_Layer':

                yoloOut.append( module( x ) )

            else:

                x = module( x )

            out.append( x if self.routs[ i ] else [] )

            if verbose:

                print( '%g/%g %s -' % (i, len( self.moduleList ), name ), list( x.shape ), str ) #type:ignore
                str = ''

        if self.training:

            return yoloOut

        elif ONNX_EXPORT:

            p = torch.cat( yoloOut, dim = 0 )

            return p

        else:

            # x : [ batchSize, -1, numClass + 5 ], p : [ batchSize, anchorNum, gridH, gridW, numClass + 5 ]
            x, p = zip( *yoloOut )
            x = torch.cat( x, 1 )

            return x, p#}}}

    def Info( self, verbose = False ):
# {{{
        np = sum( x.numel() for x in self.parameters() )
        ng = sum( x.numel() for x in self.parameters() if x.requires_grad )

        if verbose == True:

            print( '%5g, %40s, %9s, %12s, %20s, %10s, %10s' %( 'layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma' ) )

            for i, ( name, p ) in enumerate( self.named_parameters() ):

                name = name.replace( 'moduleList.', '' )
                print( '%5g, %40s, %9s, %12s, %20s, %10s, %10s' %( i, name, p.requires_grad, p.numel(), list( p.shape ), p.mean(), p.std() ) )
        try:

            from thop import profile #type:ignore
            macs, _ = profile( self, inputs = ( torch.zeros( 1, 3, 480, 480 ), ), verbose = False )
            fs = '. %.1f GFLOPS' %( macs / 1E9 * 2 )

        except:

            fs = ''

        print( 'model summary : %g layer, %g parameters, %g gradients%s' % ( len( list( self.parameters() ) ) , np, ng, fs ) )# }}}}}}

if __name__ == '__main__':

    cfg = '../Config/my_yolov3.cfg'
    torch.set_printoptions( 5 )
    darkNet = Dark_Net( cfg )

    img = torch.ones( 1, 3, 512, 512 )
    a = darkNet( img )

