from pychu import Function
from pychu.functions import broadcast_to, softmax
from pychu import utils
from pychu import gpu

###############################################################################
# 損失関数(loss function)
###############################################################################


# 平均2乗誤差
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff**2).sum() / int(diff.size)

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff*(2. / int(diff.size))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


# SoftmaxCrossEntropy関数
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        """SoftmaxCrossEntropyのforward

        Args:
            x (ndarray or Variable): 予想値, shapeは(N, 特徴量数)
            t (ndarray or Variable): 正解値, shapeは(N, )

        Returns:
            _type_: _description_

        Notation:
            N: バッチサイズ
        """
        xp = gpu.get_array_module(x)
        t_xp = gpu.to_xp(t, xp).astype(xp.int64)
        N = x.shape[0]
        log_z = utils.logsumexp(xp, axis=1)
        log_p = xp - log_z

        log_p = xp.take_along_axis(log_p, t_xp[:, None], axis=1).squeeze(1)
        scalar_N = xp.array(N, dtype=xp.float32)
        y = -log_p.sum() / scalar_N
        return y

    def backward(self, gy):
        """
        Args:
            gy (ndarray or Variable): スカラー値の勾配

        Returns:
            ndarray or Variable: 入力xに対する勾配, shapeは(x.shape)
            ndarray or Variable: ターゲットtに対する勾配, shapeは(t.shape)

        Notation:
            N: バッチサイズ
            x, t: 推論データと正解データ
        """
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = gpu.get_array_module(t.data)
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

        gx, gy = SoftmaxCrossEntropy().backward(gy)
        gx = gx.reshape(N, T, V)
        gt = gy.reshape(N, T)
        return gx, gt


def time_softmax_cross_entropy(x, t):
    return TimeSoftmaxCrossEntropy()(x, t)
