import numpy as np

from pychu.core import Function
from pychu.utils import pair, get_conv_outsize, get_deconv_outsize
from pychu import gpu


###############################################################################
# 画像変換用の関数(img translate function)
###############################################################################


# image -> columns(行列の形)
class Im2col(Function):
    def __init__(self, filter, stride, pad, to_matrix):
        """Im2colの初期化
        Args:
            filter (int or (int, int)): フィルターのサイズ
            stride (int or (int, int)): ストライドのサイズ
            pad (int or (int, int)): パディングのサイズ
            to_matrix (bool):
        """
        super().__init__()
        self.input_shape = None
        self.filter = filter
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        """
        Args:
            x (tuple): img

        Returns:
            col(list):
                to_matrix = Trueの場合は(N * OH * OW, C * FH * FW)
                to_matrix = Falseの場合は(N, C, FH, FW, OH, OW)
        """
        self.input_shape = x.shape
        y = im2col_array(x, self.filter, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        """
        Args:
            gy (Tensor, ndarray): colとして出力したもの
        forwardではim2colをしたためcol2imでimgに戻す
        """
        gx = col2im(gy, self.input_shape, self.filter, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, filter, stride=1, pad=0, to_matrix=True):
    y = Im2col(filter, stride, pad, to_matrix)(x)
    return y


def im2col_array(img, filter, stride, pad, to_matrix=True):
    """フィルタによって画像からパッチを抽出する

    Args:
        img (Tensor, ndarray): 入力画像, shapeは(N, C, H, W)
        filter (int or (int, int)): フィルタのサイズ
        stride (int or (int, int)): ストライドのサイズ
        pad (int or (int, int)): パディングのサイズ
        to_matrix (bool, optional):
            Trueなら2次元配列に変換する
            Falseなら(N, C, FH, FW, OH, OW)のまま. Defaults to True.

    Returns:
        list:
            to_matrix = Trueの場合は(N * OH * OW, C * FH * FW)
            to_matrix = Falseの場合は(N, C, FH, FW, OH, OW)

    Notation:
        N: batch size
        C: channel size
        H: image height
        W: image width
        FH: filter height
        FW: filter width
        SH: stride height
        SW: stride width
        PH: padding height
        PW: padding width
        OH: output height
        OW: output width
    """
    N, C, H, W = img.shape
    FH, FW = pair(filter)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, FH, SH, PH)
    OW = get_conv_outsize(W, FW, SW, PW)

    if gpu.gpu_backend == "cuda":
        col = _im2col_cuda(img, filter, stride, pad)
    else:
        # paddingする
        img = np.pad(img,
                     ((0, 0),  # batch軸へのpadding
                      (0, 0),  # channel軸へのpadding
                      (PH, PH + SH - 1),  # height軸へのpadding
                      (PW, PW + SW - 1)),  # width軸へのpadding
                     mode="constant", constant_values=0)
        col = np.ndarray((N, C, FH, FW, OH, OW), dtype=img.dtype)

        # バッチごとにフィルターを適用していく
        # jをフィルターのheight軸
        for j in range(FH):
            j_lim = j + SH * OH
            # iをフィルターのwidth軸
            for i in range(FW):
                i_lim = i + SW * OW
                # colのj, iの位置にimgのj:j_lim:SH, i:i_lim:SWを代入
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
    if to_matrix:
        # (N, C, FH, FW, OH, OW) -> (N, OH, OW, C, FH, FW)こうすることでreshapeでバッチごとに
        # col.shapeは(N * OH * OW, C * FH * FW)となる
        # N * OH * OWは一度のミニバッチでどれだけの値が(ピクセル数)が使えるかを表す
        # C * FH * FWは畳み込みを行うために必要な値(ピクセル)の数を表す
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, - 1))

    return col


def _im2col_cuda(img, filter, stride, pad):
    N, C, H, W = img.shape
    FH, FW = pair(filter)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, FH, SH, PH)
    OW = get_conv_outsize(W, FW, SW, PW)
    dy, dx = 1, 1
    col = gpu.xp_gpu.empty((N, C, FH, FW, OH, OW), dtype=img.dtype)

    gpu.xp_gpu.ElementwiseKernel(
        'raw T img, int32 H, int32 W, int32 OH, int32 OW,'
        'int32 FH, int32 FW, int32 SH, int32 SW, int32 PH, int32 PW,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (FH * FW * OH * OW);
           int ky = i / (FW * OH * OW) % FH;
           int kx = i / (OH * OW) % FW;
           int out_y = i / OW % OH;
           int out_x = i % OW;
           int in_y = ky * dy + out_y * SH - PH;
           int in_x = kx * dx + out_x * SW - PW;
           if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
             col = img[in_x + W * (in_y + H * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  H, W, OH, OW, FH, FW, SH, SW, PH, PW, dy, dx, col)

    return col


# columns -> image
class Col2im(Function):
    def __init__(self, input_shape, filter, stride, pad, to_matrix):
        """Col2imの初期化

        Args:
            input_shape (int or (int, int)): xのshape
            filter (int or (int, int)): フィルタのサイズ
            stride (int or (int, int)): ストライドのサイズ
            pad (int or (int, int)): パディングのサイズ
            to_matrix (bool):
        """
        super().__init__()
        self.input_shape = input_shape
        self.filter = filter
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        """
        Args:
            x (Variabel, ndarray): 入力のtensor

        Returns:
            :
        """
        y = col2im_array(x, self.input_shape, self.filter, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.filter, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, filter, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, filter, stride, pad, to_matrix)(x)


def col2im_array(col, img_shape, filter, stride, pad, to_matrix=True):
    """

    Args:
        col (Tensor, ndarray): 行列
        img_shape (int or (int, int)): 画像のshape
        filter (int or (int, int)): フィルタのサイズ
        stride (int or (int, int)): ストライドのサイズ
        pad (int or (int, int)): パディングのサイズ

        to_matrix (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Notation:
        N: batch size
        C: channel size
        H: image height
        W: image width
        FH: filter height
        FW: filter width
        SH: stride height
        SW: stride width
        PH: padding height
        PW: padding width
        OH: output height
        OW: output width
    """
    N, C, H, W = img_shape
    FH, FW = pair(filter)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, FH, SH, PH)
    OW = get_conv_outsize(W, FW, SW, PW)

    if to_matrix:
        # to_matrixがFalseのときはもとから(N, C, FH, FW, OH, OW)
        # 元々の形である(N, C, FH, FW, OH, OW)に変える
        col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    if gpu.xp_gpu == "cuda":
        img = _col2im_cuda(col, SH, SW, PH, PW, H, W)
        return img
    else:
        # colからimgに戻すときにもともと同じピクセルだったものが重複して出てきてそれを考慮するとこれだけ必要
        img = np.zeros((N, C,
                        H + 2 * PH + SH - 1,
                        W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(FH):
            j_lim = j + SH * OH
            for i in range(FW):
                i_lim = i + SW * OW

                # colのj, iの位置にimgのj:j_lim:SH, i:i_lim:SWを代入
                # colのj, iの位置を指定してimgとshapeを合わせている
                img[:, :, j:j_lim:SH, i:i_lim:SW] += \
                    col[:, :, j, i, :, :]
        # paddingを取り除く
        return img[:, :, PH:H + PH, PW:W + PW]


def _col2im_cuda(col, SH, SW, PH, PW, H, W):
    N, C, FH, FW, OH, OW = col.shape
    dy, dx = 1, 1
    img = gpu.xp_gpu.empty((N, C, H, W), dtype=col.dtype)

    gpu.xp_gpu.ElementwiseKernel(
        'raw T col, int32 H, int32 W, int32 OH, int32 OW,'
        'int32 FH, int32 FW, int32 SH, int32 SW, int32 PH, int32 PW,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (H * W);
           int y  = i / W % H;
           int x  = i % W;
           T val = 0;
           for (int ky = 0; ky < FH; ++ky) {
             int out_y = (y + PH - ky * dy);
             if (0 > out_y || out_y >= OH * SH) continue;
             if (out_y % SH != 0) continue;
             out_y /= SH;
             for (int kx = 0; kx < FW; ++kx) {
               int out_x = (x + PW - kx * dx);
               if (0 > out_x || out_x >= OW * SW) continue;
               if (out_x % SW != 0) continue;
               out_x /= SW;
               int k = out_y + OH * (kx + FW * (ky + FH * c0));
               val = val + col[out_x + OW * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  H, W, OH, OW, FH, FW, SH, SW, PH, PW, dx, dy, img)
    return img


# 畳み込み
class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        """Conv2dの初期化

        Args:
            stride (int or (int, int)): ストライドのサイズ. Defaults to 1.
            pad (int or (int, int)): パディングのサイズ. Defaults to 0.
        """
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, weight, b):
        """_summary_

        Args:
            x (Tensor or ndarray): 入力画像, shapeは(N, C, H, W)
            W (Tensor or ndarray): 重み、すべてのフィルター, shapeは(OC, C, FH, FW)
            b (Tensor or ndarray): バイアス

        Returns:
            Tensor or ndarray: 畳み込みをした4次元テンソル

        Notation:
            FH: filter height
            FW: filter width
        """
        xp = gpu.get_array_module(x)

        FH, FW = weight.shape[2:]

        # img(x)をcol((N, C, FH, FW, OH, OW)の6次元テンソル)に変換する
        col = im2col_array(x, filter=(FH, FW),
                           stride=self.stride, pad=self.pad, to_matrix=False)

        # colは(N, C, FH, FW, OH, OW), Wは(OC, C, FH, FW)のため
        # yは(N, OH, OW, OC)の4次元テンソルになる
        y = xp.tensordot(col, weight, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        # yを(N, OH, OW, OC)->(N, OC, OH, OW)にする
        y = xp.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        """
        Args:
            gy (Tensor or ndarray): forwardで返したyに対する損失関数の勾配(aL/ay)

        Returns:
            _type_: _description_
        """
        x, weight, b = self.inputs
        # deconv2dでgy(特徴マップ)をもとの画像サイズに復元している
        gx = deconv2d(gy, weight, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))

        gW = Conv2DGradW(self)(x, gy)

        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


# 転置畳み込み
class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        """Deconv2dの初期化

        Args:
            stride (int or (int, int)): ストライドのサイズ. Defaults to 1.
            pad (int or (int, int)): パディングのサイズ. Defaults to 0.
            outsize (int or (int, int)): outputのサイズ, Noneの場合は自動で計算する.
                Defaults to None
        """
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, weight, b):
        """Conv2dにおける(x)入力の勾配を求める

        Args:
            x (Tensor or ndarray): 畳み込みをされた特徴マップ
            W (Tensor or ndarray): 重さ
            b (Tensor or ndarray): バイアス

        Returns:
            Tensor or ndarray: 畳み込みをした元の画像サイズと同じ4次元テンソル

        Notation:
            N: batch size
            H: height
            W: width
            SH: stride height
            SW: stride width
            PH: pad height
            PW: pad width
            C: channel size
            OC: output channel size
            FH: filter height
            FW: filter width
            OH: output height
            OW: output width
        """
        xp = gpu.get_array_module(x)

        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, FH, FW = weight.shape

        N, C, H, W = x.shape

        if self.outsize is None:
            OH = get_deconv_outsize(H, FH, SH, PH)
            OW = get_deconv_outsize(W, FW, SW, PW)
        else:
            OH, OW = pair(self.outsize)

        img_shape = (N, OC, OH, OW)

        # Weightは(OC, C, FH, FW), xは(N, C, H, W)のため
        # gcolは(C, FH, FW, N, H, W)
        gcol = xp.tensordot(weight, x, (0, 1))

        # (C, FH, FW, N, H, W) -> (N, C, FH, FW, H, W)に変換
        gcol = xp.transpose(gcol, (3, 0, 1, 2, 4, 5))

        # col(gcol)をimg(N, C, H, W)に変換する
        y = col2im_array(gcol, img_shape, (FH, FW),
                         self.stride, self.pad, to_matrix=False)

        if b is not None:
            self.no_bias = True
            # bはOC分あるのでそれをcolのshapeに合わせる
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, gy):
        x, W, b = self.inputs

        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)

        gW = Conv2DGradW(self)(gy, x)

        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


# 畳み込み層のフィルタに対して勾配を計算する
class Conv2DGradW(Function):
    def __init__(self, conv2d):
        """Conv2DGradWの初期化

        Args:
            conv2d(Conv2D): 使用しているConv2Dのインスタンス

        Notation:
            FH: filter height
            FW: filter width
            SH: stride height
            SW: stride width
        """
        weight = conv2d.inputs[1]
        FH, FW = weight.shape[2:]
        self.filter = (FH, FW)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        """Conv2dにおける(weight)重さの勾配を求める

        Args:
            x (Tensor, ndarray): 入力画像, shapeは(N, C, H, W)
            gy (Tensor, ndarray): Conv2dで返した特徴マップ, shapeは(N, OC, OH, OW)

        Returns:
            _type_: _description_
        """
        xp = gpu.get_array_module(x)

        col = im2col_array(x, self.filter, self.stride,
                           self.pad, to_matrix=False)

        # gyは(N, OC, OH, OW), colは(N, C, FH, FW, OH, OW)
        # gWは(OC, C, FH, FW)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy
