import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
from PIL.Image import Image, fromarray


STANDARD_COLORS = [
                      'AliceBlue'     , 'Chartreuse'  , 'Aqua'          , 'Aquamarine'     , 'Azure'               , 'Beige'          , 'Bisque'          ,# {{{
                      'BlanchedAlmond', 'BlueViolet'  , 'BurlyWood'     , 'CadetBlue'      , 'AntiqueWhite'        , 'Chocolate'      , 'Coral'           ,
                      'CornflowerBlue', 'Cornsilk'    , 'Crimson'       , 'Cyan'           , 'DarkCyan'            , 'DarkGoldenRod'  , 'DarkGrey'        ,
                      'DarkKhaki'     , 'DarkOrange'  , 'DarkOrchid'    , 'DarkSalmon'     , 'DarkSeaGreen'        , 'DarkTurquoise'  , 'DarkViolet'      ,
                      'DeepPink'      , 'DeepSkyBlue' , 'DodgerBlue'    , 'FireBrick'      , 'FloralWhite'         , 'ForestGreen'    , 'Fuchsia'         ,
                      'Gainsboro'     , 'GhostWhite'  , 'Gold'          , 'GoldenRod'      , 'Salmon'              , 'Tan'            , 'HoneyDew'        ,
                      'HotPink'       , 'IndianRed'   , 'Ivory'         , 'Khaki'          , 'Lavender'            , 'LavenderBlush'  , 'LawnGreen'       ,
                      'LemonChiffon'  , 'LightBlue'   , 'LightCoral'    , 'LightCyan'      , 'LightGoldenRodYellow', 'LightGray'      , 'LightGrey'       ,
                      'LightGreen'    , 'LightPink'   , 'LightSalmon'   , 'LightSeaGreen'  , 'LightSkyBlue'        , 'LightSlateGray' , 'LightSlateGrey'  ,
                      'LightSteelBlue', 'LightYellow' , 'Lime'          , 'LimeGreen'      , 'Linen'               , 'Magenta'        , 'MediumAquaMarine',
                      'MediumOrchid'  , 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen'   , 'MediumTurquoise', 'MediumVioletRed' ,
                      'MintCream'     , 'MistyRose'   , 'Moccasin'      , 'NavajoWhite'    , 'OldLace'             , 'Olive'          , 'OliveDrab'       ,
                      'Orange'        , 'OrangeRed'   ,'Orchid'         , 'PaleGoldenRod'  , 'PaleGreen'           , 'PaleTurquoise'  , 'PaleVioletRed'   ,
                      'PapayaWhip'    , 'PeachPuff'   , 'Peru'          , 'Pink'           , 'Plum'                , 'PowderBlue'     , 'Purple'          ,
                      'Red'           , 'RosyBrown'   , 'RoyalBlue'     , 'SaddleBrown'    , 'Green'               , 'SandyBrown'     , 'SeaGreen'        ,
                      'SeaShell'      , 'Sienna'      , 'Silver'        , 'SkyBlue'        , 'SlateBlue'           , 'SlateGray'      , 'SlateGrey'       ,
                      'Snow'          , 'SpringGreen' , 'SteelBlue'     , 'GreenYellow'    , 'Teal'                , 'Thistle'        , 'Tomato'          ,
                      'Turquoise'     , 'Violet'      , 'Wheat'         , 'White'          , 'WhiteSmoke'          , 'Yellow'         , 'YellowGreen'# }}}
                  ]

def DrawText(
                draw                , box : list , cls  : int              , score    : float   ,
                categoryIndex : dict, color : str, font : str = 'arial.ttf', fontSize : int = 24
            ):
# {{{
    '''
    将目标边界框和类别信息绘制到图片上面
    '''

    try:

        font = ImageFont.truetype( font, fontSize ) #type:ignore

    except IOError:

        font = ImageFont.load_default() #type:ignore

    left, top, _, bottom = box
    displayStr = f'{categoryIndex[ str( cls ) ]} : {int( 100 * score )}%'
    displayStrHeights = [ font.getsize( ds )[ 1 ] for ds in displayStr ] #type:ignore
    displayStrHeight = ( 1 + 2 * 0.05 ) * max( displayStrHeights )

    if top > displayStrHeight:

        textTop = top - displayStrHeight
        textBottom = top

    else:

        textTop = bottom
        textBottom = bottom + displayStrHeight

    for ds in displayStr:

        textWidth, _ = font.getsize( ds ) #type:ignore
        margin = np.ceil( 0.05 * textWidth )
        draw.rectangle( [ ( left, textTop ), ( left + textWidth + 2 * margin, textBottom ) ], fill = color )
        draw.text( ( left + margin, textTop ), ds, fill = 'black', font = font )
        left += textWidth# }}}

def DrawMasks( image, masks, colors, thresh : float = 0.7, alpha : float = 0.5 ):
# {{{
    npImage = np.array( image )
    masks = np.where( masks > thresh, True, True )
    imgToDraw = np.copy( npImage )

    for mask,  color in zip( masks, colors ):

        imgToDraw[ mask ] = color

    out = npImage * ( 1 - alpha ) + imgToDraw * alpha

    return fromarray( out.astype( np.uint8 ) )# }}}


def DrawObjs(
                image : Image,
                boxes : np.ndarray = None, classes : np.ndarray = None, scores : np.ndarray = None   , #type:ignore
                masks : np.ndarray = None, categoryIndex : dict = None, boxThresh : float   = 0.1    , #type:ignore
                masksThresh : float = 0.5, lineThickness : int  = 8   , font : str = 'arial.ttf'     ,
                fontSize : int = 24      , drawBoxOnImg  : bool = True, drawMasksOnImg : bool = False,
            ):
# {{{
    '''
        将目标边界框信息, 类别信息, mask信息绘制到图片上
        Args:
            image   : 需要绘制的图片
            boxes   : 目标边界框信
            classes : 目标类别信息
            scores  : 目标概率信息
            masks   : 目标mask信息
            categoryIndex : 类别与名称字典
            boxThresh     : 过滤概率阈值
            masksThresh   : mask阈值
            lineThickness : 边界框宽度
            font     : 字体类型
            fontSize : 字体大小
            drawBoxOnImg   : 是否画box在图片上
            drawMasksOnImg : 是否画mask在图片上
    '''

    # 过滤低概率的目标
    index = np.greater( scores, boxThresh )
    boxes = boxes[ index ]
    classes = classes[ index ]
    scores = scores[ index ]

    if masks is not None:

        masks = masks[ index ]

    if len( boxes ) == 0:

        return image

    colors = [ ImageColor.getrgb( STANDARD_COLORS[ cls % len( STANDARD_COLORS ) ] ) for cls in classes ]

    if drawBoxOnImg:

        draw = ImageDraw.Draw( image )

        for box, cls, score, color in zip( boxes, classes, scores, colors ):

            left, top, right, bottom = box

            # 绘制目标边界框
            draw.line( [ ( left, top ), ( left, bottom ), ( right, bottom ), ( right, top ), ( left, top ) ], width = lineThickness, fill = color )

            # 绘制类别和概率信息
            DrawText( draw, box.tolist(), int( cls ), float( score ), categoryIndex, color, font, fontSize ) #type:ignore

    if drawMasksOnImg and ( masks is not None ):

        image = DrawMasks( image, masks, colors, masksThresh )

    return image# }}}
