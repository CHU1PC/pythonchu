import numpy as np
from typing import Any

from pychu import gpu
from pychu.core import Function, as_variable

###############################################################################
# tensor用関数(tensor function)
###############################################################################


# reshape関数
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# broadcastto関数
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = gpu.get_array_module(x)
        return xp.broadcast_to(x, self.shape)

    def backward(self, gy):
        # broadcastはデータの数を拡張しているため単純に勾配の値が増やした軸への和になる
        return sum_to(gy, self.x_shape)


def broadcast_to(x, shape):
    """xをしたいshapeに拡張する

    Args:
        x (Variable, ndarray): 変換したいndarray入力
        shape (tuple, list): 変換したい形(行列)

    Returns:
        Variable: 変換した後のVariableを返す
    """
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class Stack(Function):
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *xs):
        xp = gpu.get_array_module(xs)
        return xp.stack(xs, axis=self.axis)

    def backward(self, gy):
        return tuple(gy.take(i, axis=self.axis) for i in
                     range(gy.shape[self.axis]))


def stack(xs, axis=0):
    xs = [as_variable(x) for x in xs]
    datas = [x.data for x in xs]
    return as_variable(Stack(axis)(*datas))


# sumto関数
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        import pychu.utils as utils
        return utils.sum_to(x, self.shape)


def sum_to(x, shape):
    """xを指定したshapeになるように和をとって変形させる

    Args:
        x(Variable, ndarray): ndarrayの入力
        shape(tuple, list): 変換したい形(行列)

    Returns:
        Variable: 変換したあとのVariableを返す
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


# GetItem関数
class GetItem(Function):
    """
    get_itemで指定された配列の要素を返す
    """
    def __init__(self, slices: Any):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = gpu.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices: Any):
    f = GetItem(slices)
    return f(x)
