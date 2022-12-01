import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_Concat( nn.Module ):
    '''# {{{
        将多个特征矩阵在Channel维度进行拼接
    '''

    def __init__( self, layers ):

        super().__init__()

        self.layers = layers
        self.multiple = len( layers ) > 1

    def forward( self, _, outputs ):

        # outputs 会在正向传播中进行创建，用来记录每一层的特征的矩阵
        return torch.cat( [ outputs[ i ] for i in self.layers ], 1 ) if self.multiple else outputs[ self.layers[ 0 ] ]# }}}

class Weighted_Feature_Fusion( nn.Module ):
# {{{
    def __init__( self, layers, weight = False ):

        super().__init__()

        self.layers = layers
        self.weight = weight
        self.n = len( layers ) + 1

        if weight:

            self.w = nn.parameter.Parameter( torch.zeros( self.n ), requires_grad = True )

    def forward( self, x, outputs ):

        if self.weight:

            w = torch.sigmoid( self.w ) * ( 2 / self.n )
            x = x * w[ 0 ]

        # 融合
        nx = x.shape[ 1 ]

        for i in range( self.n - 1 ):

            a = outputs[ self.layers[ i ] ] * w[ i + 1 ] if self.weight else outputs[ self.layers[ i ] ] #type:ignore
            na = a.shape[ 1 ]

            # 根据相加的两个特征矩阵的channel选择相加的方式
            if nx == na:

                x = x + a

            elif nx > na:

                x[ :, : na ] = x[ :, : na ] + a

            else:

                x = x + a[ :, : nx ]

        return x# }}}

