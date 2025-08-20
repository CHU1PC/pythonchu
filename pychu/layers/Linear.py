# 全結合層
import numpy as np
import pychu.functions as F
from pychu import gpu
from pychu import as_tensor
from pychu.core import Parameter
from pychu.layers import Layer


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32,
                 in_size=None, W=None):
        """_summary_

        Args:
            out_size (int): 出力サイズ
            nobias (bool): Trueならばself.b=NoneとしFalseならnp.zerosで初期化.
                           Defaults to False.
            dtype (_type_): Defaults to np.float32.
            in_size (int): Noneならば自動で入力のサイズをin_sizeとする. Defaults to None.
            W (Tensor, ndarray): NoneならばParameterで初期化を行う
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        if W is not None:
            self.W = as_tensor(W)
        else:
            self.W = Parameter(None, name="W")
            if self.in_size is not None:
                self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        """W(重さ)を初期化するメソッド

        Args:
            xp (numpy or cupy): Defaults to np.
        """
        In, Out = self.in_size, self.out_size
        W_data = F.randn((In, Out), xp=xp) * np.sqrt(1 / In)
        self.W.data = W_data

    def forward(self, x):
        # in_sizeがNoneでself.WがNoneで初期化されていた場合forwardが呼び出されるときに入力と同じサイズだけ作る
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = gpu.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


# TimeLinear層
class TimeLinear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32,
                 in_size=None, W=None):
        super().__init__()
        self.in_size = in_size
        self.linear = Linear(out_size, nobias, dtype, in_size, W)
        self.out_size = out_size

    def forward(self, xs):
        """LinearのforwardをT回分行う

        Args:
            xs (Tensor, ndarray): 入力(input), shapeは(N, T, D)

        Returns:
            ys(Tensor, ndarray): 出力(output), shapeは(N, T, out_size)

        Notation:
            N: バッチサイズ(batch size)
            T: Linear層の次元数(number of Linear)
            D: 入力ベクトルの次元数(input feature dimension)  補足: 1時刻あたりの特徴量
        """
        N, T, D = xs.shape
        xs_reshape = xs.reshape(N * T, D)
        ys = self.linear(xs_reshape)

        ys = ys.reshape(N, T, self.out_size)
        return ys
