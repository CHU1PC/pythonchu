import numpy as np

import pychu.functions as F
from pychu import gpu
from pychu.core import Parameter
from pychu.layers import Layer
from pychu.utils import pair


# 畳み込み層
class Conv2d(Layer):
    def __init__(self, out_channels, filter_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter = filter_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        """W(重さ)を初期化するメソッド

        Args:
            xp (cupy or numpy): Defaults to np.

        Notation:
            C: input channels
            OC: output channels
            FH: filter height
            FW: filter width
        """
        C, OC = self.in_channels, self.out_channels
        FH, FW = pair(self.filter)

        # Xavierの初期化
        scale = np.sqrt(C * FH * FW)
        W_data = xp.random.randn(OC, C, FH, FW).astype(self.dtype) / scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = gpu.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
