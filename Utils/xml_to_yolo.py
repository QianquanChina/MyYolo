import os
import json
import shutil
from tqdm import tqdm
from lxml import etree

def GetVocPath( vocRoot, vocVersion ):
# {{{
    # 获得所需要的Voc的路径
    vocXmlPath    = os.path.join( vocRoot, vocVersion, "Annotations" )
    vocImagesPath = os.path.join( vocRoot, vocVersion, "JPEGImages" )
    valTxtPath    = os.path.join( vocRoot, vocVersion, "ImageSets", "Main", 'val.txt' )
    trainTxtPath  = os.path.join( vocRoot, vocVersion, "ImageSets", "Main", 'train.txt' )

    # 检查文件是否存在
    assert os.path.exists( vocXmlPath ), "VOC xml path not exist..."
    assert os.path.exists( vocImagesPath ), "VOC images path not exist..."
    assert os.path.exists( valTxtPath ), "VOC val txt file not exist..."
    assert os.path.exists( trainTxtPath ), "VOC train txt file not exist..."

    return trainTxtPath, valTxtPath, vocXmlPath, vocImagesPath# }}}

def ParseXmlToDict( xml ):
# {{{
    '''
        将xml文件解析成字典的格式
    '''

    if len( xml ) == 0:

        return { xml.tag : xml.text }

    result = {}

    for child in xml:

        # 使用递归遍历标签的信息
        childResult = ParseXmlToDict( child )

        if child.tag != 'object':

            result[ child.tag ] = childResult[ child.tag ] #type:ignore

        else:

            if child.tag not in result:

                result[ child.tag ] = []

            result[ child.tag ].append( childResult[ child.tag ] ) #type:ignore

    return { xml.tag : result }# }}}

def TranslateInfo( fileNames, saveRoot, vocImagesPath, vocXmlPath, classDict, trainVal = 'train' ):
# {{{
    saveTxtPath = os.path.join( saveRoot, trainVal, 'labels' )

    if os.path.exists( saveTxtPath ) is False:

        os.makedirs( saveTxtPath )

    saveImgPath = os.path.join( saveRoot, trainVal, 'images' )

    if os.path.exists( saveImgPath ) is False:

        os.makedirs( saveImgPath )

    for file in tqdm( fileNames, desc = 'translat {} file...'.format( trainVal ) ):

        # 检查图像文件是否存在
        imgPath = os.path.join( vocImagesPath, file + '.jpg' )
        assert os.path.exists( imgPath ), 'file : {} not exist...'.format( imgPath )

        # 检查xml文件是否存在
        xmlPath = os.path.join( vocXmlPath, file + '.xml' )
        assert os.path.exists( xmlPath ), 'file : {} not exist...'.format( xmlPath )

        # 读取xml文件
        with open( xmlPath ) as fid:

            xmlStr = fid.read()

        xml = etree.fromstring( xmlStr )
        data = ParseXmlToDict( xml )[ 'annotation' ]

        # 获得图像的高宽用于计算目标的相对坐标
        imgHeight = int( data[ 'size' ][ 'height' ] )
        imgWidth = int( data[ 'size' ][ 'width' ] )

        # 将目标的信息写入到txt中
        assert 'object' in data.keys(), "file : '{}' lack of object key".format( xmlPath )

        if len( data[ 'object' ] ) == 0:

            # 此时xml文件中不存在目标就直接忽略掉该样本
            print( "Warning : '{}' xml, there are not objects".format( xmlPath ) )
            continue

        with open( os.path.join( saveTxtPath, file + '.txt' ), 'w' ) as f:

            # 遍历图像中的所有目标
            for index, obj in enumerate( data[ 'object' ] ):

                className = obj[ 'name' ]
                classIndex = classDict[ className ] - 1
                xmin = float( obj[ 'bndbox' ][ 'xmin' ] )
                xmax = float( obj[ 'bndbox' ][ 'xmax' ] )
                ymin = float( obj[ 'bndbox' ][ 'ymin' ] )
                ymax = float( obj[ 'bndbox' ][ 'ymax' ] )

                # 进一步检查数据，有的标注信息中有w、h为0的情况，这样数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:

                    print( "Warning : in '{}' xml, there are some bbox w/h <= 0".format( xmlPath ) )
                    continue

                # 将box信息转换成yolo格式
                xCenter = xmin + ( xmax - xmin ) / 2
                yCenter = ymin + ( ymax - ymin ) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 将绝对坐标转换成相对坐标，保存小数后六位
                xCenter = round( xCenter/ imgWidth, 6 )
                yCenter = round( yCenter / imgHeight, 6 )
                w = round( w / imgWidth, 6 )
                h = round( h / imgHeight, 6 )

                info = [ str( i ) for i in [ classIndex, xCenter, yCenter, w, h ] ]

                if index == 0:

                    f.write( ' '.join( info ) )

                else:

                    f.write( '\n' + ' '.join( info ) )

        # 将图片拷贝到对应的目录下面
        pathCopyTo = os.path.join( saveImgPath, imgPath.split( os.sep )[ -1 ] )

        if os.path.exists( pathCopyTo ) is False:

            shutil.copyfile( imgPath, pathCopyTo )# }}}

def creatClassNames( classDict ):
# {{{
    keys = classDict.keys()

    with open( '../Data/my_data_label.names', 'w' ) as w:

        for index, k in enumerate( keys ):

            if index + 1 == len( keys ):

                w.write( k )

            else:

                w.write( k + '\n' )# }}}

if __name__ == '__main__':
# {{{
    vocRoot = '/home/jc/Study/DeepLearn/Pytorch/Net/SegmentationNetwork/DataSet/VOCdevkit/'
    vocVersion = 'VOC2012'

    saveFileRoot = '../Data/DataSet'

    if os.path.exists( saveFileRoot ) is False:

        os.makedirs( saveFileRoot )

    assert os.path.exists( saveFileRoot ), "save file dir does not exist..."

    # 这个文件需要提前写好
    labelJsonPath = '../Data/pascal_voc_classes.json'
    assert os.path.exists( labelJsonPath ), "label_json_path does not exist..."

    # 获得所需要的Voc的路径
    trainTxtPath, valTxtPath, vocXmlPath, vocImagesPath = GetVocPath( vocRoot, vocVersion )

    jsonFile = open( labelJsonPath, 'r' )
    classDict = json.load( jsonFile )

    # 读取train.txt的所有行信息，删除空行
    with open( trainTxtPath, 'r' ) as r:

        trainFileNames = [ i for i in r.read().splitlines() if len( i.strip() ) > 0 ]

    # voc的信息转换成yolo，并将图片复制到相应的文件夹下面
    TranslateInfo( trainFileNames, saveFileRoot, vocImagesPath, vocXmlPath, classDict, 'train' )

    # 读取val.txt的所有行信息，删除空行
    with open( valTxtPath, 'r' ) as r:

        valFileNames = [ i for i in r.read().splitlines() if len( i.strip() ) > 0 ]

    # voc的信息转换成yolo，并将图片复制到相应的文件夹下面
    TranslateInfo( valFileNames, saveFileRoot, vocImagesPath, vocXmlPath, classDict, 'val' )

    # 创建my_data_lable.names文件
    creatClassNames( classDict )# }}}
