import numpy as np


from pychu import Function
from pychu.functions import broadcast_to, softmax
from pychu import utils
from pychu import cuda

###############################################################################
# 損失関数(loss function)
###############################################################################


# 平均2乗誤差
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff**2).sum() / len(diff)

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff*(2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


# SoftmaxCrossEntropy関数
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


# TimeSoftmaxCrossEntropy関数
class TimeSoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        """TimeSoftmaxCrossEntropyのforward

        Args:
            x (ndarray or Variable): 入力系列, shapeは(N, T, V)
            t (ndarray or Variable): ターゲット系列, shapeは(N, T)

        Returns:
            SoftmaxCrossEntropyの出力, スカラー値

        Notation:
            N: バッチサイズ
            T: シーケンス長
            V: 語彙数(入力系列の次元数)
        """
        N, T, V = x.shape

        # SoftmaxCrossEntropyの入力に合わせるためにreshape
        # xは(N, T, V)でシーケンス長をバッチサイズにmultiplyして, あらゆる時刻における語句の確率分布を計算する
        x = x.reshape(N * T, V)
        t = t.reshape(N * T)

        loss = SoftmaxCrossEntropy().forward(x, t)
        return loss

    def backward(self, gy):
        x, t = self.inputs
        N, T, V = x.shape

        grad = SoftmaxCrossEntropy().backward(gy)
        gx = grad[0].reshape(N, T, V)
        gt = grad[1].reshape(N, T)
        return gx, gt


def time_softmax_cross_entropy(x, t):
    return TimeSoftmaxCrossEntropy()(x, t)
