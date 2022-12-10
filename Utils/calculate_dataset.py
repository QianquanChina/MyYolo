'''
1.获得训练集和验证集图片的路径并放在对应的.txt文件中
2.创建data.data文件，记录classes的个数，以及1中生成的.txt的路径和my_data_lable.name（由xml_to_yolo.py生成）文件路径
3.根据yolov3-spp.cfg创建my_yolov3.cfg文件修改其中的predictor、filter以及yolo classes参数（这两个参数是根据类别数改变的）

'''

import os

def CaculateDataTxt( txtPath, dataSetDir ):
# {{{
    # 创建my_train_(val)data.txt用来记录图片的路径AnnotationDir
    with open( txtPath, 'w' ) as w:

        for fileName in os.listdir( dataSetDir ):

            if fileName == 'classes.txt':

                continue

            imgPath = os.path.join( dataSetDir.replace( 'labels', 'images' ), fileName.split( '.' )[ 0 ] + '.jpg' )
            line = imgPath + '\n'
            assert os.path.exists( imgPath ), 'file : {} not exist'.format( imgPath )
            w.write( line )# }}}

def CreateDataData( createDataPath, labelPath, trainPath, valPath, classesInfo ):
# {{{
    # 创建my_data.data 文件用来记录类别个数、my_val(train)_data.txt、my_data_label.names的路径

    with open( createDataPath, 'w' ) as w:

        # 记录类别个数
        w.write( 'classes = {}'.format( len( classesInfo ) ) + '\n' )

        # 记录my_train_data.txt路径
        w.write( 'train = {}'.format( trainPath ) + '\n' )

        # 记录my_val_data.txt路径
        w.write( 'val = {}'.format( valPath ) + '\n' )

        # 记录my_data_labels.names路径
        w.write( 'names = {}'.format( labelPath ) + '\n' )# }}}

def ChangeAndCreateCfgFile( cfgPath, classesInfo, saveCfgPath = '../Config/my_yolov3.cfg' ):
# {{{
    filtersLines = [ 636, 722, 809 ]
    classesLines = [ 643, 729, 816 ]

    cfgLines = open( cfgPath, 'r' ).readlines()

    for i in filtersLines:

        assert 'filters' in cfgLines[ i - 1 ], 'filters param is not in line : {}'.format( i - 1 )
        outputNum = ( 5 + len( classesInfo ) ) * 3
        cfgLines[ i - 1 ] = 'filters={}\n'.format( outputNum )

    for i in classesLines:

        assert 'classes' in cfgLines[ i - 1 ], 'classes param is not in line:{}'.format( i - 1 )
        cfgLines[ i - 1 ] = 'classes={}\n'.format( len(classesInfo ) )

    with open( saveCfgPath, 'w' ) as w:

        w.writelines( cfgLines )# }}}

if __name__ == '__main__':
# {{{
    rootPath = os.path.dirname( os.path.dirname( os.path.realpath( __file__ ) ) )
    cfgPath = '../Config/yolov3-spp.cfg'
    classesLabel = os.path.join( rootPath, 'Data/my_data_label.names' )
    valAnnotationDir = os.path.join( rootPath, 'Data/DataSet/val/labels' )
    trainAnnotationDir = os.path.join( rootPath, 'Data/DataSet/train/labels' )

    assert os.path.exists( cfgPath ), 'cfgPath not exist'
    assert os.path.exists( classesLabel ), 'classesLabel not exist'
    assert os.path.exists( valAnnotationDir ), 'valAnnotationDir not exist'
    assert os.path.exists( trainAnnotationDir ), 'trainAnnotationDir not exist'

    # 统计训练集和验证集的数据并生成对应的.txt文件
    valTxtPath = os.path.join( rootPath, 'Data/my_val_data.txt' )
    trainTxtPath = os.path.join( rootPath, 'Data/my_train_data.txt' )

    CaculateDataTxt( trainTxtPath, trainAnnotationDir )
    CaculateDataTxt( valTxtPath, valAnnotationDir )

    classesInfo = [ line.strip() for line in open( classesLabel, 'r' ).readlines() if len( line.strip() ) > 0 ]

    # 创建my_data.data 文件用来记录类别个数、my_val(train)_data.txt、my_data_label.names的路径
    CreateDataData( '../Data/my_data.data', classesLabel, trainTxtPath, valTxtPath, classesInfo )

    # 根据yolov3-spp.cfg创建my_yolov3.cfg文件修改其中的predictor、filters以及yolo classes参数（这两个参数是根据类别数改变的）
    ChangeAndCreateCfgFile( cfgPath, classesInfo )# }}}
