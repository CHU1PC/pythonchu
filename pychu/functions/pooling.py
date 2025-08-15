import numpy as np

from pychu.core import Function
from pychu.utils import pair
from pychu.functions import im2col_array, col2im_array
from pychu import gpu


class Pooling(Function):
    def __init__(self, filter, stride=1, pad=0):
        super().__init__()
        self.filter = filter
        self.stride = stride
        self.pad = pad

    def forward(self, x):

        col = im2col_array(x, self.filter, self.stride, self.pad,
                           to_matrix=False)
        # N: batch size
        # C: channel size
        # FH: filter height
        # FW: filter width
        # OH: output height
        # OW: output width
        N, C, FH, FW, OH, OW = col.shape

        col = col.reshape(N, C, FH * FW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.filter = mpool2d.filter
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = gpu.get_array_module(gy)

        # N: batch size
        # C: channel size
        # OH: output height
        # OW: output width
        N, C, OH, OW = gy.shape

        # H: height
        # W: width
        N, C, H, W = self.input_shape

        # FH: filter height
        # FW: filter width
        FH, FW = pair(self.filter)

        gcol = xp.zeros((N * C * OH * OW * FH * FW), dtype=self.dtype)

        indexes = (self.indexes.ravel() +
                   xp.arange(0, self.indexes.size * FH * FW, FH * FW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, FH, FW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.filter, self.stride,
                          self.pad, to_matrix=False)

        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.filter = mpool2d.filter
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.filter, self.stride, self.pad,
                           to_matrix=False)
        # N: batch size
        # C: channel size
        # FH: filter height
        # FW: filter width
        # OH: output height
        # OW: output width
        N, C, FH, FW, OH, OW = col.shape
        col = col.reshape(N, C, FH * FW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, FH * FW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, filter, stride=1, pad=0):
    return Pooling(filter, stride, pad)(x)
