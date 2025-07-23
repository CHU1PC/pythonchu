import numpy as np

from pychu import Function
from pychu.functions import sum_to, broadcast_to
from pychu import cuda


###############################################################################
# Variable用の数学関数(math function for Variable)
###############################################################################


# matmul関数
class MatMul(Function):
    def forward(self, x, W):
        W = cuda.as_gpu_array(W)
        y = x.dot(W) if hasattr(x, "dot") else x @ W
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    """行列積を計算するもの

    Args:
        x (Variable, ndarray): テンソル
        W (Variable, ndarray): x・WのW

    Returns:
        Variable : 行列積をした後の行列を返す
    """
    return MatMul()(x, W)


# Linear関数
class Linear(Function):
    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)
        W = cuda.to_xp(W, xp)
        y = x.dot(W) if hasattr(x, "dot") else x @ W

        if b is not None:
            b = cuda.to_xp(b, xp)
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# transpose関数
class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        gx = transpose(gy)
        return gx


def transpose(x):
    return Transpose()(x)


# sum関数
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


# sin関数
class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


def sin(x):
    return Sin()(x)


# cos関数
class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.cos(x)

    def backward(self, gy):
        x, = self.inputs
        return -gy * sin(x)


def cos(x):
    return Cos()(x)


# exp**x関数
class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)
