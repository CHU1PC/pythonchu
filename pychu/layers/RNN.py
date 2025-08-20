from pychu import gpu
from pychu import as_tensor
from pychu.layers import Layer
from pychu.layers import Linear
import pychu.functions as F


# RNN層
class RNN(Layer):
    """RNN層を作っている

    RNN層は時系列データを取り扱えるが、逆伝番においてtanhの勾配は1-y^2でありあらゆる点で1-y^2 <= 1のため勾配消失が起こってしまう
    ほかにも(Linear層の中で)Matmulを使っているため勾配爆発や勾配消滅が起きやすい(行列の時これは特異値のよって決まる, 特異値はWが重さの時
    W @ WTの固有値の平方根(固有値は |W @ WT - tE| = 0 となるtの値)で求めれるこれの最大値が1以下ならば勾配爆発はしない)

    補足: hidden(隠れ状態)とは出力に相当する, また短期的な情報を保持する
    """
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)

    def reset_state(self):
        self.prev_h = None

    def forward(self, x, h=None):
        if h is None:
            # 以前の隠れ状態がなければそのままinputからhiddenを求める
            h_new = F.tanh(self.x2h(x))
        else:
            # 以前の隠れ状態がある場合それと新しく作ったhiddenをLinearに通して和をとる
            h_new = F.tanh(self.x2h(x) + self.h2h(h))
        return h_new


# TimeRNN層
class TimeRNN(Layer):
    def __init__(self, hidden_size, in_size=None, stateful=False):
        """TimeRNNの初期化

        Args:
            hidden_size (int): 中間層の次元数
            in_size (int): 入力サイズ, Noneならば入力のsizeを自動でin_sizeとする.
                           Defaults to None.
            stateful (bool): TrueであればTimeRNNのなかの各RNN同士が隠れ状態を引き継ぐ.
                             Defaults to False.
        """
        super().__init__()
        self.rnn_cell = RNN(hidden_size, in_size)
        self.hidden_size = hidden_size
        self.stateful = stateful
        self.reset_state()

    def reset_state(self):
        self.prev_hidden = None

    def forward(self, xs):
        """RNNのforwardをT回分行う

        Args:
            xs (Tensor, ndarray): 入力(input), shapeは(N, T, D)

        Returns:
            hs(Tensor, ndarray): 出力(output), shapeは(N, T, hidden_size)

        Notation:
            N: バッチサイズ(batch size)
            T: RNNの次元数(number of RNN)
            D: 入力ベクトルの次元数(input feature dimension)  補足: 1時刻あたりの特徴量
        """
        N, T, D = xs.shape
        hs = []
        xp = gpu.get_array_module(xs)
        xs = xp.array(xs)
        # prev_hiddenが存在してstateful = Trueならばhを以前の隠れ状態とする
        h = self.prev_hidden if self.stateful and self.prev_hidden is not None\
            else xp.zeros((N, self.hidden_size), dtype=xs.dtype)

        for t in range(T):
            x = xs[:, t, :]
            # hがあればそれをprev_hとしてRNNのほうに保存する
            # ここでrnnを以前の隠れ状態に加えて一回分適応したものがhとなる
            h = self.rnn_cell(x, h)
            hs.append(as_tensor(h))

        hs = F.stack(hs, axis=1)

        if self.stateful:
            # statefulがTrueのときは以前の時系列データを受け継ぐ
            self.prev_hidden = h
        return hs
