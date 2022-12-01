import copy
import json
import torch
import pickle
import torchvision
import numpy as np
import torch.utils.data
import torch.distributed as dist
from tqdm import tqdm
from pycocotools.coco import COCO
from collections import defaultdict
from pycocotools.cocoeval import COCOeval

def ConvertToCocoApi( ds ):
# {{{
    cocoDs = COCO()

    # annotation的IDS是从1开始的
    annId = 1
    dataSet = { 'images' : [], 'categories' : [], 'annotations' : [] }

    # 用于记录检测目标所属的类别，防止多次记录同一个数据所以使用set类型
    categories = set()

    # 遍历dataSet中的每张图片
    for imgIdx in tqdm( range( len( ds ) ), desc = 'loading eval info for coco tools.' ):

        # imgIdx : 记录这是dataSet中的第几张图片
        # targets.shape : [ numObj, 5 ], 5 : [ objIdx, x, y, w, h ], shapes : [ h, w ]
        targets, shapes = ds.CocoIndex( imgIdx )
        imgDict = {}
        imgDict[ 'id' ] = imgIdx
        imgDict[ 'height' ] = shapes[ 0 ]
        imgDict[ 'width' ] = shapes[ 1 ]
        dataSet[ 'images' ].append( imgDict )

        for obj in targets:

            ann = {}
            ann[ 'image_id' ] = imgIdx

            # 将相对坐标转换为绝对坐标
            boxes = obj[ 1 : ]

            # boxes[ x, y, w, h ] ---- [ xMin, yMin, w, h ]
            boxes[ : 2 ] -= 0.5 * boxes[ 2 : ]
            boxes[ [ 0, 2 ] ] *= imgDict[ 'width'  ]
            boxes[ [ 1, 3 ] ] *= imgDict[ 'height' ]

            # 将Tensor变成List
            boxes = boxes.tolist()

            ann[ 'bbox' ] = boxes

            # 记录该目标属于那个检测类别的id
            ann[ 'category_id' ] = int( obj[ 0 ] )
            categories.add( int( obj[ 0 ] ) )
            ann[ 'area' ] = boxes[ 2 ] * boxes[ 3 ]
            ann[ 'iscrowd' ] = 0

            # 图片中每个标记被标记物体的id, 即该图片有多少个目标
            ann[ 'id' ] = annId
            dataSet[ 'annotations' ].append( ann )
            annId += 1

    dataSet[ 'categories' ] = [
                                  { 'id' : i }
                                  for i in sorted( categories )
                              ]
    cocoDs.dataset = dataSet
    cocoDs.createIndex()

    return cocoDs# }}}

def GetCocApiFromDataSet( dataSet ):
# {{{
    for _ in range( 10 ):

        if isinstance( dataSet, torchvision.datasets.CocoDetection ):

            break

        if isinstance( dataSet, torch.utils.data.Subset ):

            dataSet = dataSet.dataset

    if isinstance( dataSet, torchvision.datasets.CocoCaptions ):

        return dataSet.coco

    return ConvertToCocoApi( dataSet )# }}}

def CreateIndex( self ):
# {{{
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict( list ), defaultdict( list )

    if 'annotations' in self.dataset:

        for ann in self.dataset[ 'annotations' ]:

            imgToAnns[ ann[ 'image_id' ] ].append( ann )
            anns[ ann[ 'id' ] ] = ann

    if 'images' in self.dataset:

        for img in self.dataset[ 'images' ]:

            imgs[ img[ 'id' ] ] = img

    if 'categories' in self.dataset:

        for cat in self.dataset[ 'categories' ]:

            cats[ cat[ 'id' ] ] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:

        for ann in self.dataset[ 'annotations' ]:

            catToImgs[ ann[ 'category_id' ] ].append( ann[ 'image_id' ] )

    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats# }}}

def LoadRes( self, resFile ):
# {{{
    res = COCO()
    res.dataset[ 'images' ] = [
                                  img
                                  for img in self.dataset[ 'images' ]
                              ]

    if isinstance( resFile, torch._six.string_classes ):

        anns = json.load( open( resFile ) )

    elif type( resFile ) == np.ndarray:

        anns = self.loadNumpyAnnotations( resFile )

    else:

        anns = resFile

    assert type( anns ) == list, 'results in not an array of objects'

    annsImgIds = [
                     ann[ 'image_id' ]
                     for ann in anns
                 ]
    assert set( annsImgIds ) == ( set( annsImgIds ) & set( self.getImgIds() ) ), 'Results do not correspond to current coco set'

    if 'caption' in anns[ 0 ]:

        imgIds = set(
                        [
                            img[ 'id' ]
                            for img in res.dataset[ 'images' ]
                        ]

                    ) & \
                 set(
                        [
                            ann[ 'image_id' ]
                            for ann in anns
                        ]
                    )
        res.dataset[ 'images' ] = [
                                      img
                                      for img in res.dataset[ 'images' ]
                                      if img[ 'id' ] in imgIds
                                  ]

        for id, ann in enumerate( anns ):

            ann[ 'id' ] = id + 1

    elif 'bbox' in anns[ 0 ] and not anns[ 0 ][ 'bbox' ] == []:

        res.dataset[ 'categories' ] = copy.deepcopy( self.dataset[ 'categories' ] )

        for id, ann in enumerate( anns ):

            bb = ann[ 'bbox' ]
            x1, x2, y1, y2 = [ bb[ 0 ], bb[ 0 ] + bb[ 2 ], bb[ 1 ], bb[ 1 ] + bb[ 3 ] ]

            if 'segmentation' not in ann:

                ann[ 'segmentation' ] = [ [ x1, y1, x1, y2, x2, y2, x2, y1 ] ]

            ann[ 'area' ] = bb[ 2 ] * bb[ 3 ]
            ann[ 'id' ] = id + 1
            ann[ 'iscrowd' ] = 0

    res.dataset[ 'annotations' ] = anns
    CreateIndex( res )

    return res# }}}

def Evaluate( self ):
# {{{
    p = self.params

    if p.useSegm is not None:

        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print( 'useSegm(deprecated) is not None. Running {} evaluation'.format( p.iouType ) )

    p.imgIds = list( np.unique( p.imgIds ) )

    if p.useCats:

        p.catIds = list( np.unique( p.catIds ) )

    p.maxDets = sorted( p.maxDets )
    self.params = p
    self._prepare()

    catIds = p.catIds if p.useCats else [ -1 ]

    if p.iouType == 'segm' or p.iouType == 'bbox':

        computeIoU = self.computeIoU

    elif p.iouType == 'keypoints':

        computeIoU = self.computeOks

    self.ious = {

                    ( imgId, catId ) : computeIoU(imgId, catId) #type:ignore
                    for imgId in p.imgIds
                    for catId in catIds
                }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[ -1 ]
    evalImgs = [
                  evaluateImg( imgId, catId, areaRng, maxDet )
                  for catId in catIds
                  for areaRng in p.areaRng
                  for imgId in p.imgIds
               ]
    evalImgs = np.asarray( evalImgs ).reshape( len( catIds ), len( p.areaRng ), len( p.imgIds ) )
    self._paramsEval = copy.deepcopy( self.params )

    return p.imgIds, evalImgs# }}}

def IsDistAvailAndInitialized():
# {{{
    """检查是否支持分布式环境"""
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

def AllGather( data ):
# {{{
    worldSize = GetWorldSize()

    if worldSize == 1:

        return [ data ]

    # 序列化张量
    buffer = pickle.dumps( data )
    storage = torch.ByteStorage.from_buffer( buffer )
    tensor = torch.ByteTensor( storage ).to( 'cuda' )

    localSize = torch.tensor( [ tensor.numel() ], device = 'cuda' )
    sizeList = [ torch.tensor( [ 0 ], device = 'cuda') for _ in range( worldSize ) ]
    dist.all_gather( sizeList, localSize )
    sizeList = [ int( size.item() ) for size in sizeList ]
    maxSize = max( sizeList )

    tensorList = []

    for _ in sizeList:

        tensorList.append( torch.empty( (maxSize, ), dtype = torch.uint8, device = 'cuda' ) )

    if localSize != maxSize:

        padding = torch.empty( size = ( maxSize - localSize, ), dtype = torch.uint8, device = 'cuda' ) #type:ignore
        tensor = torch.cat( ( tensor, padding ), dim = 0 )

    dist.all_gather( tensorList, tensor )

    dataList = []

    for size, tensor in zip( sizeList, tensorList ):

        buffer = tensor.cpu().numpy().tobytes()[ :size ]
        dataList.append( pickle.loads( buffer ) )

    return dataList# }}}

def Merge( imgIds, evalImgs ):
# {{{
    allImgIds = AllGather( imgIds )
    allEvalImgs = AllGather( evalImgs )

    mergedImgIds = []

    for p in allImgIds:

        mergedImgIds.extend( p )

    mergedEvalImgs = []

    for p in allEvalImgs:

        mergedEvalImgs.append( p )

    mergedImgIds = np.array(mergedImgIds)
    mergedEvalImgs = np.concatenate( mergedEvalImgs, 2 )

    mergedImgIds, idx = np.unique( mergedImgIds, return_index=True )
    mergedEvalImgs = mergedEvalImgs[ ..., idx ]

    return mergedImgIds, mergedEvalImgs# }}}

def CreateCommonCocoEval( cocoEval, imgIds, evalImgs ):
# {{{
    imgIds, evalImgs = Merge( imgIds, evalImgs )
    imgIds = list( imgIds )
    evalImgs = list( evalImgs.flatten() )

    cocoEval.evalImgs = evalImgs
    cocoEval.params.imgIds = imgIds
    cocoEval._paramsEval = copy.deepcopy( cocoEval.params )# }}}

class Coc_Evaluator( object ):
# {{{
    def __init__( self, cocoGt, iouTypes ):
# {{{
        assert isinstance( iouTypes, ( list, tuple ) )
        cocoGt = copy.deepcopy( cocoGt )
        self.cocoGt = cocoGt
        self.iouTypes = iouTypes
        self.cocoEval = {}

        for iouType in iouTypes:

            self.cocoEval[ iouType ] = COCOeval( cocoGt, iouType = iouType )

        self.imgIds = []
        self.evalImgs = {
                            k : []
                            for k in iouTypes
                        }# }}}

    def Update( self, predictions ):
# {{{
        imgIds = list( np.unique( list( predictions.keys() ) ) )
        self.imgIds.extend( imgIds )

        for iouType in self.iouTypes:

            results = self.Prepare( predictions, iouType )
            cocoDt = LoadRes( self.cocoGt, results ) if results else COCO()
            cocoEval = self.cocoEval[ iouType ]
            cocoEval.cocoDt = cocoDt
            cocoEval.params.imgIds = list( imgIds )
            imgIds, evalImgs = Evaluate( cocoEval )
            self.evalImgs[ iouType ].append( evalImgs )# }}}

    def SynchronizeBetweenProcesses( self ):
# {{{
        for iouType in self.iouTypes:

            self.evalImgs[ iouType ] = np.concatenate( self.evalImgs[ iouType ], 2 ) #type:ignore
            CreateCommonCocoEval( self.cocoEval[ iouType ], self.imgIds, self.evalImgs[ iouType ] )# }}}

    def Accumulate( self ):
# {{{
        for cocoEval in self.cocoEval.values():

            cocoEval.accumulate()# }}}

    def Summarize( self ):
# {{{
        for iouType, cocoEval in self.cocoEval.items():

            print( 'IoU metric : {}'.format( iouType ) )
            cocoEval.summarize()# }}}

    def Prepare( self, predictions, iouType ):
# {{{
        if iouType == 'bbox':

            return self.PrepareForCocoDetection( predictions )
        else:

            raise ValueError( 'Unkonwn iou type {}'.format( iouType ) )# }}}

    def PrepareForCocoDetection( self, predictions ):
# {{{
        cocoResults = []

        for originalId, prediction in predictions.items():

            if len( prediction ) == 0:

                continue

            boxes = prediction[ 'boxes' ]
            xMin, yMin, xMax, yMax = boxes.unbind( 1 )
            boxes = torch.stack( ( xMin, yMin, xMax - xMin, yMax - yMin ), dim = 1 ).tolist()
            scores = prediction[ 'scores' ].tolist()
            labels = prediction[ 'labels' ].tolist()
            cocoResults.extend(
                                  [
                                      {
                                          'image_id' : originalId,
                                          'category_id' : labels[ k ],
                                          'bbox' : box,
                                          'score' : scores[ k ]
                                      }
                                      for k, box in enumerate( boxes )
                                  ]
                              )

        return cocoResults# }}}}}}

