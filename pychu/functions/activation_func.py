import numpy as np

from pychu import Function
from pychu import gpu


# tanh関数
class Tanh(Function):
    def forward(self, x):
        xp = gpu.get_array_module(x)
        return xp.tanh(x)

    def backward(self, gy):
        # tanhの微分は1 - f(x)**2のためtanhの出力結果が必要となる
        # またoutputsはweekref(弱参照)のため()がいる
        y = self.outputs[0]()
        return gy * (1 - y**2)  # type: ignore


def tanh(x):
    return Tanh()(x)


# sigmoid関数
class Sigmoid(Function):
    def forward(self, x):
        xp = gpu.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)  # type: ignore
        return gx


def sigmoid(x):
    return Sigmoid()(x)


# softmax関数
class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = gpu.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backeward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


# ReLU関数
class ReLU(Function):
    def forward(self, x):
        return np.maximum(x, 0.0)

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)
