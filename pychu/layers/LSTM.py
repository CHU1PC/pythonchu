import pychu.functions as F
from pychu import cuda
from pychu.layers import Layer
from pychu import as_variable
from pychu.layers import Linear


# LSTM層
class LSTM(Layer):
    """LSTM層を作っている

    LSTM層ではRNN層の弱点であった勾配爆発や勾配消滅が起きてしまうところを改善している, LSTMでは勾配クリッピング(勾配のノルムで勾配を割って
    そこに定数(勾配のノルムの最大値)をかける、[例: |a| >= c -> a = c * a / |a|^2], ほかにもゲートを3つつけることで
    より勾配消失を防ぐことができる

    補足: cell(セル状態)とは、ある時刻tにおけるLSTMの長期記憶(過去の情報)を保持する内部状態
    """
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        HID, IN = hidden_size, in_size
        self.x2f = Linear(HID, in_size=IN)
        self.x2i = Linear(HID, in_size=IN)
        self.x2o = Linear(HID, in_size=IN)
        self.x2u = Linear(HID, in_size=IN)
        self.h2f = Linear(HID, in_size=HID, nobias=True)
        self.h2i = Linear(HID, in_size=HID, nobias=True)
        self.h2o = Linear(HID, in_size=HID, nobias=True)
        self.h2u = Linear(HID, in_size=HID, nobias=True)
        self.reset_state()  # reset_state()でprev_hiddenとprev_cellの初期化を行う

    def reset_state(self):
        # 今までの隠れ状態とセル状態を初期化する
        self.prev_hidden = None
        self.prev_cell = None

    def forward(self, x):
        if self.prev_hidden is None:
            # 隠れ状態がない場合
            # forget(忘却)ゲート
            forget = F.sigmoid(self.x2f(x))
            # input(入力)ゲート
            input = F.sigmoid(self.x2i(x))
            # output(出力)ゲート
            output = F.sigmoid(self.x2o(x))
            # cellは新しく覚える情報
            cell = F.tanh(self.x2u(x))

        else:
            # 隠れ状態がすでにある場合
            # forget(忘却)ゲート
            forget = F.sigmoid(self.x2f(x) + self.h2f(self.prev_hidden))
            # input(入力)ゲート
            input = F.sigmoid(self.x2i(x) + self.h2i(self.prev_hidden))
            # output(出力)ゲート
            output = F.sigmoid(self.x2o(x) + self.h2o(self.prev_hidden))
            # cellは新しく覚える情報
            cell = F.tanh(self.x2u(x) + self.h2u(self.prev_hidden))

        if self.prev_cell is None:
            cell_new = (input * cell)
        else:
            cell_new = (forget * self.prev_cell) + (input * cell)

        hidden_new = output * F.tanh(cell_new)

        self.prev_hidden, self.prev_cell = hidden_new, cell_new
        return hidden_new


class TimeLSTM(Layer):
    def __init__(self, hidden_size, in_size=None, stateful=False):
        """TimeLSTMの初期化

        Args:
            hidden_size (int): 中間層の次元数
            in_size (int): 入力サイズ, Noneならば入力のsizeを自動でin_sizeとする.
                           Defaults to None.
            stateful (bool): TrueであればTimeRNNのなかの各LSTM同士が隠れ状態を引き継ぐ.
                             Defaults to False.
        """
        super().__init__()
        self.lstm_cell = LSTM(hidden_size, in_size)
        self.hidden_size = hidden_size
        self.stateful = stateful
        self.reset_state()  # reset_state()でprev_hiddenとprev_cellを初期化する

    def reset_state(self):
        # 今までの隠れ状態とセル状態を初期化する
        self.prev_hidden = None
        self.prev_cell = None

    def forward(self, xs):
        """LSTMのforwardをT回分行う

        Args:
            xs (Variable, ndarray): 入力(input), shapeは(N, T, D)

        Returns:
            hs(Variable, ndarray): 出力(output), shapeは(N, T, hidden_size)

        Notation:
            N: バッチサイズ(batch size)
            T: LSTMの次元数(number of RNN)
            D: 入力ベクトルの次元数(input feature dimension)  補足: 1時刻あたりの特徴量
        """
        N, T, D = xs.shape
        hs = []
        xp = cuda.get_array_module(xs)
        h = self.prev_hidden if self.stateful and self.prev_hidden is not None\
            else xp.zeros((N, self.hidden_size), dtype=xs.dtype)
        c = self.prev_cell if self.stateful and self.prev_cell is not None \
            else xp.zeros((N, self.hidden_size), dtype=xs.dtype)
        self.lstm_cell.prev_hidden = h
        self.lstm_cell.prev_cell = c

        for t in range(T):
            x = xs[:, t, :]
            h = self.lstm_cell(x)
            c = self.lstm_cell.prev_cell
            hs.append(as_variable(h))

        hs = F.stack(hs, axis=1)

        if self.stateful:
            self.prev_hidden = h
            self.prev_cell = c
        return hs
