from pychu import gpu
from pychu.layers import Layer


class Embedding(Layer):
    def __init__(self, W):
        """Embedding層の初期化

        Args:
            W (Tensor or ndarray): 埋め込み行列, shapeは(V, D)でVは語彙数, Dは埋め込みベクトルの次元数
        """
        super().__init__()
        self.W = W
        self.idx = None

    def forward(self, idx):
        """ idxに対応する埋め込みベクトルを返す

        Args:
            idx (number.Number or list, ndarray, Tensor):
                埋め込みベクトルを取得したいインデックス, shapeは(N, )でNはバッチサイズ
        Returns:
            Tensor or ndarray: 埋め込みベクトル, shapeは(N, D)
        """
        self.idx = idx
        return self.W[idx]

    def backward(self, dout):
        """ 勾配を計算する

        Args:
            dout (Tensor or ndarray):
                埋め込みベクトルに対する勾配, shapeは(N, D)でNはバッチサイズ, Dは埋め込みベクトルの次元数

        Returns:
            _type_: _description_
        """
        xp = gpu.get_array_module(dout)
        dW = xp.zeros_like(self.W)
        # dWのidxの位置にdoutを足す
        xp.add.at(dW, self.idx, dout)  # type: ignore
        return dW


class TimeEmbedding(Layer):
    def __init__(self, W):
        """ TimeEmbedding層の初期化

        Args:
            W (Tensor or ndarray): 埋め込み行列, shapeは(V, D)でVは語彙数, Dは埋め込みベクトルの次元数
        """
        super().__init__()
        self.W = W
        self.layers: list[Embedding] = []

    def forward(self, xs):
        """ Embedding層のforwardをT回分行う

        Args:
            xs (Tensor, ndarray): 入力(input), shapeは(N, T)でNはバッチサイズ, Tは時刻数

        Returns:
            out (Tensor, ndarray): 出力(output), shapeは(N, T, D)でDは埋め込みベクトルの次元数

        Notation:
            N: バッチサイズ(batch size)
            T: シーケンス長(number of time steps)
            D: 埋め込みベクトルの次元数(embedding vector dimension)  補足: 1時刻あたりの特徴量
            V: 語彙数(vocabulary size)
        """
        N, T = xs.shape
        V, D = self.W.shape
        xp = gpu.get_array_module(self.W.data)
        # Embedding層の出力である(N, D)をT回適応した形にする
        if gpu.gpu_backend == "metal":
            dtype = gpu.canonical_dtype(self.W.dtype, xp)
            out = xp.zeros((N, T, D), dtype=dtype)
        else:
            out = xp.zeros((N, T, D), dtype=self.W.dtype)

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    # 特殊な勾配処理が必要なためbackwardを実装
    def backward(self, dout):
        N, T, D = dout.shape
        xp = gpu.get_array_module(dout)
        grad = xp.zeros_like(self.W)
        for t in range(T):
            grad += self.layers[t].backward(dout[:, t, :])
        return grad
