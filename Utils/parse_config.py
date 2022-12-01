import os
import numpy as np

def ParseModelCfg( path ):
# {{{
    # 检查文件是否存在
    if not path.endswith( '.cfg' ) or not os.path.exists( path ):

        raise FileNotFoundError( 'the cfg file not exist...' )

    # 读取文件信息
    with open( path, 'r' ) as f:

        lines = f.read().split( '\n' )

    # 去除空行和注释行
    lines = [ x for x in lines if x and not x.startswith( '#' ) ]

    # 去除每行开头和结尾的空格符
    lines = [ x.strip() for x in lines ]

    # 存储构建模型所需要的信息
    modelCfg = []

    for line in lines:

        if line.startswith( '[' ):

            # 如果检查到是新的一部分，则给此处创建一个字典
            modelCfg.append( {} )
            modelCfg[ -1 ][ 'type' ] = line[ 1 : -1 ].strip()

            if modelCfg[ -1 ][ 'type' ] == 'convolutional':

                modelCfg[ -1 ][ 'batch_normalize' ] = 0

        else:

            key, val = line.split( '=' )
            key = key.strip()
            val = val.strip()

            if key == 'anchors':

                val = val.replace( ' ', '' )
                modelCfg[ -1 ][ key ] = np.array( [ float( x ) for x in val.split( ',' ) ] ).reshape( ( -1, 2 ) )

            elif ( key in [ 'from', 'layers', 'mask' ] or ( key == 'size' and ',' in val ) ):

                modelCfg[ -1 ][ key ] = [ int( x ) for x in val.split( ',' ) ]

            else:

                if val.isnumeric():

                    modelCfg[ -1 ][ key ] = int( val ) if ( int( val ) - float( val ) ) == 0  else float( val )

                else:

                    modelCfg[ -1 ][ key ] = val

    supported = [
                    'type'      , 'batch_normalize', 'filters'       , 'size'                 , 'stride'       , 'pad'        ,
                    'activation', 'layers'         , 'groups'        , 'from'                 , 'mask'         , 'anchors'    ,
                    'classes'   , 'num'            , 'jitter'        , 'ignore_thresh'        , 'truth_thresh' , 'random'     ,
                    'stride_x'  , 'stride_y'       , 'weights_type'  , 'weights_normalization', 'scale_x_y'    , 'beta_nms'   ,
                    'nms_kind'  , 'iou_loss'       , 'iou_normalizer', 'cls_normalizer'       , 'iou_thresh'   , 'probability',
                ]

    for x in modelCfg[ 1: ]:

        for k in x:

            if k not in supported:

                raise ValueError( 'Unsupported fileds:{} in cfg'.format( k ) )

    return modelCfg# }}}

def ParseDataCfg( path ):
# {{{
    # 解析配置文件
    if not os.path.exists( path ) and os.path.exists( '../Data' + os.sep + path ):

        path = 'data' + os.sep + path

    with open( path, 'r' ) as f:

        lines = f.readlines()

    options = dict()

    for line in lines:

        line = line.strip()

        if line == '' or line.startswith( '#' ):

            continue

        key, val = line.split( '=' )
        options[ key.strip() ] = val.strip()

    return options# }}}

if __name__ == '__main__':

    modelCfg = ParseModelCfg( './test.cfg' )
    print( modelCfg )

