import os
import cv2
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ExifTags
from torch.utils.data import Dataset

# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
# {{{
    if ExifTags.TAGS[ orientation ] == "Orientation":

        break# }}}

def ExifSize( img ):
# {{{
    '''
        获取图像的高宽
    '''

    s = img.size

    try:

        rotation = dict( img._getexif().items() )[ orientation ]
        if rotation == 6:  # rotation 270  顺时针翻转90度

            s = ( s[ 1 ], s[ 0 ] )

        elif rotation == 8:  # ratation 90  逆时针翻转90度

            s = ( s[ 1 ], s[ 0 ] )
    except:

        # 如果图像的exif信息中没有旋转信息，则跳过
        pass

    return s# }}}

def LoadImage( self, index ):
# {{{
    path = self.imgFiles[ index ]
    img = cv2.imread( path )
    assert img is not None, 'Image not found' + path
    h0, w0 = img.shape[ : 2 ]

    #imgSize的设置是预处理后输出的图片尺寸
    r = self.imgSize / max( h0, w0 )

    if r != 1:

        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR

        # 进行等比例缩放
        img = cv2.resize( img, ( int( w0 * r ), int( h0 * r ) ), interpolation = interp )

    # img, originalHw, resizeHw
    return img, ( h0, w0 ), img.shape[ : 2 ]# }}}

def LoadMosaic( self, index ):
# {{{
    '''
        将四张图片拼接在一张马赛克图像中
    '''
    # 拼接图像的lable信息
    labels4 = []
    s = self.imgSize

    # 随机初始化拼接图像的中心点坐标
    xc, yc = [ int( random.uniform( s * 0.5, s * 1.5 ) ) for _ in range( 2 ) ]

    # 从dataset中寻找三张图片进行拼接
    indices = [ index ] + [ random.randint( 0, len( self.labels ) - 1 ) for _ in range( 3 ) ]

    # 遍历四张图像进行拼接
    for i, index in enumerate( indices ):

        img, _, ( h, w ) = LoadImage( self, index )

        # 左上角
        if i == 0:

            # 创建马赛克图像
            img4 = np.full( ( s * 2, s * 2, img.shape[ 2 ] ), 114, dtype = np.uint8 )

            # 计算马赛克图片中的坐标信息（将图片填充到马赛克图像中）xMin, yMin, xMax, yMax
            x1a, y1a, x2a, y2a = max( xc - w, 0 ), max( yc - h, 0 ), xc, yc

            # 计算截取图像区域信息
            x1b, y1b, x2b, y2b = w - ( x2a - x1a ), h - ( y2a - y1a ), w, h

        # 右上角
        elif i == 1:

            x1a, y1a, x2a, y2a = xc, max( yc - h, 0 ), min( xc + w, s * 2 ), yc
            x1b, y1b, x2b, y2b = 0, h - ( y2a - y1a ), min( w, x2a - x1a ), h

        # 左下角
        elif i == 2:

            x1a, y1a, x2a, y2a = max( xc - w, 0 ), yc, xc, min( s * 2, yc + h )
            x1b, y1b, x2b, y2b = w - ( x2a -x1a ), 0, max( xc, w ), min( y2a - y1a, h )

        # 右下角
        elif i == 3:

            x1a, y1a, x2a, y2a = xc, yc, min( xc + w, s * 2), min( s * 2, yc + h )
            x1b, y1b, x2b, y2b = 0, 0, min( w, x2a - x1a ), min( y2a - y1a, h )

        # 将截取的图像区域填充到马赛克的相应位置
        img4[ y1a : y2a, x1a : x2a ] = img[ y1b : y2b, x1b : x2b ] #type:ignore

        # 计算pad（图像边界与马赛克的距离，越界的情况为负）
        padW = x1a - x1b #type:ignore
        padH = y1a - y1b #type:ignore

        # 获取labels对应的信息,即一张图片中有多少个目标
        # x.shape : [ obj, 5 ], 5 : [ clsIndex, xCenter, yCenter, w, h ]
        x = self.labels[ index ]
        labels = x.copy()

        # 计算标注数据在马赛克中的坐标
        if x.size > 0:

            # xMin
            labels[ :, 1 ] =  w * ( x[ :, 1 ] - x[ :, 3 ] / 2 ) + padW

            # yMin
            labels[ :, 2 ] =  h * ( x[ :, 2 ] - x[ :, 4 ] / 2 ) + padH

            # xMax
            labels[ :, 3 ] =  w * ( x[ :, 1 ] + x[ :, 3 ] / 2 ) + padW

            # yMax
            labels[ :, 4 ] =  h * ( x[ :, 2 ] + x[ :, 4 ] / 2 ) + padH

        labels4.append( labels )

    if len( labels4 ):

        labels4 = np.concatenate( labels4, 0 )

        # 设置上下限防止越界
        np.clip( labels4[ :, 1 : ], 0, 2 * s, out = labels4[ :, 1 : ] ) #type:ignore

    # 数据增强（随机旋转、缩放、平移、错切）
    img4, labels4 = RandomAffine(
                                    img4, labels4, degrees = self.hyp[ 'degrees' ], #type:ignore
                                    translate = self.hyp[ 'translate' ], scale  = self.hyp[ 'scale' ],
                                    shear     = self.hyp[ 'shear'     ], border = -s // 2
                                )

    return img4, labels4# }}}

def RandomAffine( img, targets = (), degrees = 10, translate = .1, scale = .1, shear = 10, border = 0 ):
# {{{
    # 最终输出的图片尺寸，等于img4.shape / 2
    height = img.shape[ 0 ] + border * 2
    width  = img.shape[ 1 ] + border * 2

    # 生成旋转以及缩放矩阵
    # 生成对角矩阵
    R = np.eye( 3 )

    # 随机旋转角度
    a = random.uniform( -degrees, degrees )

    # 随机缩放因子
    s = random.uniform( 1 - scale, 1 + scale )
    R[ : 2 ] = cv2.getRotationMatrix2D( angle = a, center = ( img.shape[ 1 ] / 2, img.shape[ 0 ] / 2 ), scale = s )

    # 生成平移矩阵
    T = np.eye( 3 )
    T [ 0, 2 ] = random.uniform( -translate, translate ) * img.shape[ 0 ] + border
    T [ 1, 2 ] = random.uniform( -translate, translate ) * img.shape[ 1 ] + border

    # 生成错切矩阵
    S = np.eye( 3 )
    S[ 0, 1 ] = math.tan( random.uniform( -shear, shear ) * math.pi / 180 )
    S[ 1, 0 ] = math.tan( random.uniform( -shear, shear ) * math.pi / 180 )

    # 生成一个总矩阵
    M = S @ T @ R

    if ( border != 0 ) or ( M != np.eye( 3 ) ).any():

        # 进行仿射变化
        img = cv2.warpAffine( img, M[ : 2 ], dsize = ( width, height ), flags = cv2.INTER_LINEAR, borderValue = ( 114, 114, 114 ) )

    # 对目标边界框进行仿射变换
    n = len( targets )

    if n:

        xy = np.ones( ( n * 4, 3 ) )

        # 得到边界框的四个坐标
        xy[ :, : 2 ] = targets[ :, [ 1, 2, 3, 4, 1, 4, 3, 2 ] ].reshape( n * 4, 2 )
        xy = ( xy @ M.T )[ :, : 2 ].reshape( n, 8 )

        # 对transformer后的box进行修正（假设变化后的bbox变成了菱形，此时要修正成矩形）
        x = xy[ :, [ 0, 2, 4, 6 ] ] #type:ignore
        y = xy[ :, [ 1, 3, 5, 7 ] ] #type:ignore
        xy = np.concatenate( ( x.min( 1 ), y.min( 1 ), x.max( 1 ), y.max( 1 ) ) ).reshape( 4, n ).T

        # 对坐标进行裁剪，防止越界
        xy[ :, [ 0, 2 ] ] = xy[ :, [ 0, 2 ] ].clip( 0, width  ) #type:ignore
        xy[ :, [ 1, 3 ] ] = xy[ :, [ 1, 3 ] ].clip( 0, height ) #type:ignore
        w = xy[ :, 2 ] - xy[ :, 0 ] #type:ignore
        h = xy[ :, 3 ] - xy[ :, 1 ] #type:ignore

        # 计算调整后每个box的面积
        area = w * h

        # 计算调整前每个box的面积
        area0 = ( targets[ :, 3 ] - targets[ :, 1 ] ) * ( targets[ :, 4 ] - targets[ :, 2 ] )

        # 计算每个box的比例
        ar = np.maximum( w / ( h + 1e-16 ), h / ( w + 1e-16 ) )

        # 选取长宽大于4个像素，且调整前后的面积比例大于0.2且box的高宽比例小于10的box
        i = ( w > 4 ) & ( h > 4 ) & ( area / ( area0 * s + 1e-16) > 0.2 ) & ( ar < 10 )
        targets = targets[ i ]
        targets[ :, 1 : 5 ] = xy[ i ]

    return img, targets# }}}

def LetterBox( img, newShape = ( 416, 416 ), color = ( 114, 114, 114 ), auto = True, scaleFill = False, scaleUp = True ):
# {{{
    shape = img.shape[ : 2 ]

    if isinstance( newShape, int ):

        newShape = ( newShape, newShape )

    # 缩放比例
    r = min( newShape[ 0 ] / shape[ 0 ], newShape[ 1 ] / shape[ 1 ] )

    # 对于大于指定输入大小的图片进行缩放，小于的不变
    if not scaleUp:

        r = min( r, 1.0 )

    # 计算padding
    ratio = r, r
    newUnPad = int( round( shape[ 1 ] * r ) ), int( round( shape[ 0 ] * r ) )
    dw, dh = newShape[ 1 ] - newUnPad[ 0 ], newShape[ 0 ] - newUnPad[ 1 ]

    # 保证原图比例不变，将图像最大边缩放到指定大小
    if auto:

        # 取余操作保证padding后的图片是32的整数倍
        dw, dh = np.mod( dw, 32 ), np.mod( dh, 32 )

    # 直接将图片缩放到所指定的尺寸
    elif scaleFill:

        dw, dh = 0, 0
        newUnPad = newShape
        ratio = newShape[ 0 ] / shape[ 1 ], newShape[ 1 ] / shape[ 0 ]

    # 将padding分到上下和左右两侧
    dw /= 2
    dh /= 2

    if shape[ :: -1 ] != newUnPad:

        img = cv2.resize( img, newUnPad, interpolation = cv2.INTER_LINEAR )

    top , bottom = int( round( dh - 0.1 ) ), int( round( dh + 0.1 ) )
    left, right  = int( round( dw - 0.1 ) ), int( round( dw + 0.1 ) )
    img = cv2.copyMakeBorder( img, top,bottom, left, right, cv2.BORDER_CONSTANT, value = color )

    return img, ratio, ( dw, dh )# }}}

def AugmentHsv( img, hGain = 0.5, sGain = 0.5, vGain = 0.5 ):
# {{{
    r = np.random.uniform( -1, 1, 3 ) * [ hGain, sGain, vGain ] + 1
    h, s, v = cv2.split( cv2.cvtColor( img, cv2.COLOR_BGR2HSV ) )
    dtype = img.dtype
    x = np.arange( 0, 256, dtype = np.int16 )
    lutH = ( ( x * r[ 0 ] % 180 ) ).astype( dtype )
    lutS = np.clip( x * r[ 1 ], 0, 255 ).astype( dtype )
    lutV = np.clip( x * r[ 2 ], 0, 255 ).astype( dtype )

    imgHsv = cv2.merge( ( cv2.LUT( h, lutH ), cv2.LUT( s, lutS ), cv2.LUT( v, lutV ) ) ).astype( dtype )
    cv2.cvtColor( imgHsv, cv2.COLOR_HSV2BGR, dst = img )# }}}

class Data_Set( Dataset ):
# {{{
    def __init__(
                    self           ,
                    path           , imgSize   = 416  , batchSize   = 4  ,
                    hyp     = None , rect      = None , pad         = 0.0,
                    augment = False, singleCls = False, rank        = -1 ,
                ):
# {{{
        '''
            path : 训练集或者验证集的存放图片地址的txt文件的路径
        '''

        super().__init__()

        try:

            path = str( Path( path ) )

            if os.path.isfile( path ):

                with open( path, 'r' ) as f:

                    # 读取每一行图片的路径信息
                    f = f.read().splitlines()
            else:

                raise Exception( '%s does not exist' % path )

            # 检查每张图片的路径是否在所支持的范围内
            self.imgFormats = [ '.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng' ]
            self.imgFiles = [ x for x in f if os.path.splitext( x )[ -1 ].lower() in self.imgFormats ]
            self.imgFiles.sort()

        except Exception as e:

            raise FileNotFoundError( 'Error loading data from {}. {}'.format( path, e ) )

        # 如果图片列表没有则报错
        n = len( self.imgFiles )
        assert n > 0, 'No images found in %s.' % ( path )

        # 将数据划分到一个个batch中
        bi = np.floor( np.arange( n ) / batchSize ).astype( int )

        # 记录数据集划分后的总batch数
        nb = bi[ -1 ] + 1

        # 图片的总数量
        self.n = n

        # 记录每一张图片属于哪一个batch( if batch is 4, self.batch = [ 0, 0, 0, 0, 1, 1, 1, 1 ... ] )
        self.batch = bi

        # 预处理后输出的图片尺寸
        self.imgSize = imgSize

        # 是否启用augmentHsv
        self.augment = augment

        # 超参数字典，包含图片增强所需要的参数
        self.hyp = hyp

        # 是否启用rectangular training
        self.rect = rect

        # 开启rect之后，mosaic就默认关闭
        self.mosaic = self.augment and not self.rect

        # 获取图像对应的label路径
        self.labelFiles = [ x.replace( 'images', 'labels').replace( os.path.splitext( x )[ -1 ], '.txt' ) for x in self.imgFiles ]

        # 读取每一张图片的高宽
        sp = path.replace( '.txt', '.shapes' )

        try:

            with open( sp, 'r' ) as f:

                s = [ x.split() for x in f.read().splitlines() ]

                # 判断现有的shape文件中的行数（图片个数）是否与当前数据集中图像的个数相等，如果不等则认为是不同的数据集然后重新生成.shapes文件
                assert len( s ) == n, 'shape file out of aync'
        except Exception as e:

            if rank in [ -1, 0 ]:

                imgFiles = tqdm( self.imgFiles, desc = 'Reading image shapes' )

            else:

                imgFiles = self.imgFiles

            s = [ ExifSize( Image.open( f ) ) for f in imgFiles ]
            np.savetxt( sp, s, fmt = '%g' ) #type:ignore

        # 记录每张图片的原始尺寸
        self.shapes = np.array( s, dtype = np.float64 )

        # 如果为true，训练网络时会对图片进行等比例缩放，让最长边变成imgSize
        # Note : 开启rect之后，mosaic就默认关闭，训练的时候是不开启rect操作
        if self.rect:

            s = self.shapes

            # 计算每个图片的高/宽比
            ar = s[ :, 1 ] / s[ :, 0 ]

            # 从小到大排序获得对应原图上的索引
            irect = ar.argsort()

            # 根据排序顺序重新设置图片顺序、标签顺序以及shape顺序
            self.imgFiles = [ self.imgFiles[ i ] for i in irect ]
            self.labelFiles = [ self.labelFiles[ i ] for i in irect ]
            self.shapes = s[ irect ]
            ar = ar[ irect ]

            # 计算每个batch采用的同一尺度[ h, w ]
            shapes = [ [ 1, 1 ] ] * nb

            for i in range( nb ):

                ari = ar[ bi == i ]

                # 获得第i个batch中，最小和最大的高宽比
                minI, maxI = ari.min(), ari.max()

                # 如果高/宽小于1(w>h)，将w设置为imgSize
                if maxI < 1:

                    shapes[ i ] = [ maxI, 1 ]

                elif minI > 1:

                    shapes[ i ] = [ 1, 1 / minI ]

            self.batchShapes = np.ceil( np.array( shapes ) * imgSize / 32. + pad ).astype( int ) * 32

        # label: [ class, x, y, w, h ]其中的xywh都是相对值
        self.labels = [ np.zeros( ( 0, 5 ), dtype = np.float32 ) ] * n
        labelsLoaded = False

        # nm：缺少标签的数据, nf：找到了多少数据, ne：多少个标签是空的, nd：标签中多少个是重复的
        nm, nf, ne, nd = 0, 0, 0, 0

        # 当rect为True时会对self.labels进行从新排序
        # 这里分别命名是为了防止出现rect为False/True时混淆用导致计算mAP错误
        if rect is True:

            npLabelsPath = str( Path( self.labelFiles[ 0 ]).parent ) + '.rect.npy'

        else:

            npLabelsPath = str( Path( self.labelFiles[ 0 ]).parent ) + '.norect.npy'

        if os.path.isfile( npLabelsPath ):

            x = np.load( npLabelsPath, allow_pickle = True )

            if len( x ) == n:

                # 如果载入缓存标签个数与当前计算图像数目相同则认为是同一数据集，直接读取缓存数据
                self.labels = x
                labelsLoaded = True

        if rank in [ -1, 0 ]:

            pBar = tqdm( self.labelFiles )

        else:

            pBar = self.labelFiles

        # 遍历载入标签文件
        for i, file in enumerate( pBar ):

            if labelsLoaded is True:

                l = self.labels[ i ]

            else:

                # 从文件读取标签信息
                try:

                    with open( file, 'r' ) as f:

                        # 读取每一行label，并按空格划分数据
                        l = np.array( [ x.split() for x in f.read().splitlines() ], dtype = np.float32 )

                except Exception as e:

                    print( 'An erro occurred while loading the file {} : {}'.format( file, e ) )
                    nm += 1
                    continue

            if l.shape[ 0 ]:

                # 标签的信息每行必须是五个数值[ class, x, y, w, h ]
                assert l.shape[ 1 ] == 5, '> 5 label columns : %s' % file
                assert ( l >= 0 ).all(), 'negative labels : %s' % file
                assert( l[ :, 1 : ] <= 1 ).all(), 'non-normalized or out of bounds coordinate labels : %s' %file

                # 检查每一行，看是否有重复的信息
                if np.unique( l, axis = 0 ).shape[ 0 ] < l.shape[ 0 ]:

                    nd += 1

                if singleCls:

                    l[ :, 0 ] = 0

                self.labels[ i ] = l
                nf += 1

            else:

                ne += 1

            if rank in [ -1, 0 ]:

                # 更新进度描述信息
                pBar.desc = 'Caching labels ( %g found, %g missing, %g empty, %g duplicate, for %g images )' % ( nf, nm, ne, nd, n ) # type:ignore

        assert nf > 0, 'No labels found in %s.' % os.path.dirname( self.labelFiles[ 0 ] ) + os.sep

        # 如果标签信息没有保存成numpy的格式，且训练样本数大于1000则将标签保存程numpy格式
        if not labelsLoaded and n > 1000:

            print( 'Saving labels to %s for faster future loading' % npLabelsPath )
            np.save( npLabelsPath, self.labels )# }}}

    def __len__( self ):
# {{{
        return len( self.imgFiles )# }}}

    def __getitem__( self, index ):
# {{{
        hyp = self.hyp

        if self.mosaic:

            img, labels = LoadMosaic( self, index ) #type:ignore
            shapes = None

        else:

            # 加载图片
            # h0, w0 是图片的原始高宽, h, w 是经过缩放到指定大小的高宽
            img, ( h0, w0 ), ( h, w ) = LoadImage( self, index )
            shape = self.batchShapes[ self.batch[ index ] ] if self.rect else self.imgSize
            img, ratio, pad = LetterBox( img, shape, auto = False, scaleUp = self.augment )
            shapes = ( h0, w0 ), ( ( h / h0, w / w0 ), pad )

            # 加载标签
            labels = []
            x = self.labels[ index ]

            # 对目标边界框进行相应的缩放和填充保证边界框的正确性
            if x.size > 0:

                # labels : [ class, x, y, w, h ]
                labels = x.copy()

                # labels : [ class, x, y, w, h ] ---- [ class, xMin, yMin, Xmax, Ymax ]（得到绝对坐标）
                labels[ :, 1 ] = ratio[ 0 ] * w * ( x[ :, 1 ] - x[ :, 3 ] / 2 ) + pad[ 0 ]
                labels[ :, 2 ] = ratio[ 1 ] * h * ( x[ :, 2 ] - x[ :, 4 ] / 2 ) + pad[ 1 ]
                labels[ :, 3 ] = ratio[ 0 ] * w * ( x[ :, 1 ] + x[ :, 3 ] / 2 ) + pad[ 0 ]
                labels[ :, 4 ] = ratio[ 1 ] * h * ( x[ :, 2 ] + x[ :, 4 ] / 2 ) + pad[ 1 ]

        if self.augment:

            if not self.mosaic:

                # 数据增强( 随机旋转、缩放、平移、错切 )
                img, labels = RandomAffine(
                                              img, labels,
                                              degrees   = self.hyp[ 'degrees'   ], #type:ignore
                                              translate = self.hyp[ 'translate' ], #type:ignore
                                              scale     = self.hyp[ 'scale'     ], #type:ignore
                                              shear     = self.hyp[ 'shear'     ]  #type:ignore
                                          )

            AugmentHsv( img, hGain = hyp[ 'hsv_h' ], sGain = hyp[ 'hsv_s' ], vGain = hyp[ 'hsv_v' ] ) #type:ignore

        nL = len( labels )

        if nL:

            # labels : [ class, xMin, yMin, Xmax, Ymax ] ---- [ class, x, y, w, h ]
            x = labels.copy()
            labels[ :, 1 ] = ( x[ :, 1 ] + x[ :, 3 ] ) / 2 #type:ignore
            labels[ :, 2 ] = ( x[ :, 2 ] + x[ :, 4 ] ) / 2 #type:ignore
            labels[ :, 3 ] = ( x[ :, 3 ] - x[ :, 1 ] )     #type:ignore
            labels[ :, 4 ] = ( x[ :, 4 ] - x[ :, 2 ] )     #type:ignore

            # 将绝对坐标变成相对坐标
            labels[ :, [ 2, 4 ] ] /= img.shape[ 0 ] #type:ignore
            labels[ :, [ 1, 3 ] ] /= img.shape[ 1 ] #type:ignore

        if self.augment:

            # 随机水平反转
            lrFlip = True

            if lrFlip and random.random() < 0.5:

                img = np.fliplr( img )

                if nL:

                    labels[ :, 1 ] = 1 - labels[ :, 1 ] #type:ignore

            # 随机上下反转
            udFlip = False

            if udFlip and random.random() < 0.5:

                img = np.flipud( img )

                if nL:

                    labels[ : 2 ] = 1 - labels[ :, 2 ] #type:ignore

        # nL : number Labels 即图片中有多少个目标
        labelsOut = torch.zeros( ( nL, 6 ) )

        if nL:

            labelsOut[ :, 1 : ] = torch.from_numpy( labels )

        # BGR ---- RGB, HWC ---- CHW
        img = img[ :, :, :: -1 ].transpose( 2, 0, 1 )

        # 进行了transpose所以内存不连续了,需要变成内存连续.
        img = np.ascontiguousarray( img )

        return torch.from_numpy( img ), labelsOut, self.imgFiles[ index ], shapes, index# }}}

    def CocoIndex( self, index ):
# {{{
        '''
            该方法专门为cocotools统计标签信息准备，不对图像和标签做任何处理
        '''

        # wh ---- hw
        Oshape = self.shapes[ index ][ :: -1 ]

        # 加载标签
        x = self.labels[ index ]

        # labels : class, x, y, w, h
        labels = x.copy()

        return torch.from_numpy( labels ), Oshape# }}}

    @staticmethod
    def collate_fn( batch ):
# {{{
        img, label, path, shapes, index = zip( *batch )

        for i, l in enumerate( label ):

            l[ :, 0 ] = i

        return torch.stack( img, 0 ), torch.cat( label, 0 ), path, shapes, index# }}}}}}

if __name__ == '__main__':

    import yaml
    hypFile = open( '../Config/hyp.yaml', 'r' )
    hyp = yaml.load( hypFile, yaml.FullLoader )
    hypFile.close()
    dataSet = Data_Set(
                          path = '../Data/my_train_data.txt', rect = False,
                          hyp = hyp
                      )
