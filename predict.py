import os
import cv2
import time
import json
import torch
import pylab
import numpy as np
from PIL import Image
from Backbone import Dark_Net
from Utils import LetterBox, ScaleCoords, Nms, DrawObjs

def Main():
# {{{
    imgSize = 512
    cfgPath = './Config/my_yolov3.cfg'
    imgPath = './2.jpg'
    jsonPath = './Data/pascal_voc_classes.json'
    weightsPath = './Weights/best.pt'
    assert os.path.exists( cfgPath     ), 'cfg file {} dose not exists.'.format( cfgPath )
    assert os.path.exists( weightsPath ), 'weights file {} dose not exists.'.format( weightsPath )
    assert os.path.exists( jsonPath    ), 'json file {} dose not exists.'.format( jsonPath )
    assert os.path.exists( imgPath     ), 'img file {} dose not exists.'.format( imgPath )

    with open( jsonPath, 'r' ) as f:

        classDict = json.load( f )

    categoryIndex = { str( v ) : str( k ) for k, v in classDict.items() }
    inputSize = ( imgSize, imgSize )
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
    model = Dark_Net( cfgPath, imgSize )
    weightsDict = torch.load( weightsPath, map_location = 'cpu' )
    weightsDict = weightsDict[ 'model' ] if 'model' in weightsDict else weightsDict
    model.load_state_dict( weightsDict )
    model.to( device )

    model.eval()

    with torch.no_grad():

        img = torch.zeros( ( 1, 3, imgSize, imgSize ), device = device )
        model( img )

        imgOrigin = cv2.imread( imgPath ) #type:ignore
        assert imgOrigin is not None, 'image not found ' + imgPath

        img = LetterBox( imgOrigin, newShape = inputSize, auto = True, color = ( 0, 0, 0 ) )[ 0 ]
        img = img[ :, :, :: -1 ].transpose( 2, 0, 1 )
        img = np.ascontiguousarray( img )
        img = torch.from_numpy( img ).to( device ).float()
        img /= 255
        img = img.unsqueeze( 0 )
        t1 = time.time()
        pred = model( img )[ 0 ]
        t2 = time.time()
        print( 'model spend time : ', t2 - t1 )
        pred = Nms( pred, confThres = 0.1, iouThres = 0.6, mutiLabel = True )[ 0 ]
        t3 = time.time()
        print( 'nms spend time : ', t3 - t2 )

        if pred is None:

            print( 'No target detected' )
            exit( 0 )

        pred[ :, : 4 ] = ScaleCoords( img.shape[ 2 : ], pred[ :, : 4 ], imgOrigin.shape ).round()
        print( pred.shape )

        bboxes = pred[ :, : 4 ].detach().cpu().numpy()
        scores = pred[ :, 4 ].detach().cpu().numpy()
        classes = pred[ :, 5 ].detach().cpu().numpy().astype( np.int0 ) + 1

        pilImg = Image.fromarray( imgOrigin[ :, :, :: -1 ] )

        plotImag = DrawObjs(
                               pilImg, bboxes, classes, scores,
                               categoryIndex = categoryIndex  ,
                               boxThresh     = 0.2            ,
                               lineThickness = 3              ,
                               font          = 'arial.ttf'    ,
                               fontSize      = 20
                           )

        pylab.imshow( plotImag )
        pylab.show()# }}}

if __name__ == '__main__':

    Main()
