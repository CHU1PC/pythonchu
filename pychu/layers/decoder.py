import numpy as np
from pychu.layers import Layer, TimeEmbedding, TimeLSTM, TimeLinear


class Decoder(Layer):
    def __init__(self, input_size, embedding_dim, hidden_size,
                 num_layers=1, padding_idx=None):
        """Decoderで使われるレイヤーを初期化する

        Args:
            input_size (int): 入力言語の語句数(vocab_size)
            embedding_dim (int): 埋め込みベクトルの次元数
            hidden_size (int): LSTMレイヤーの隱れ層の次元数
            num_layers (int): LSTMレイヤーの層数. Defaults to 1.
            padding_idx (int): paddingのindexを与える. Defaults to None.
        """
        super().__init__()
        self.padding_idx = padding_idx

        # Embedding層
        embed_W = np.random.randn(input_size,
                                  embedding_dim).astype(np.float32) * 0.01
        self.embed = TimeEmbedding(embed_W)

        # LSTM層
        self.lstm_layers = []
        for i in range(num_layers):
            in_size = embedding_dim if i == 0 else hidden_size
            lstm = TimeLSTM(hidden_size, in_size)
            setattr(self, f"lstm_{i}", lstm)
            self.lstm_layers.append(lstm)

        # Linear層
        self.fc = TimeLinear(input_size)  # 出力サイズは入力サイズと同じ(語句数分のベクトルを返せばいい)

    def reset_state(self):
        for lstm in self.lstm_layers:
            lstm.reset_state()

    def forward(self, xs, h0=None):
        """

        Args:
            xs (ndarray or Variable): 入力系列(時系列データ), shapeは(N, T)
            h0 (_type_, optional): Encoderなどで得た隱れ状態. Defaults to None.

        Returns:
            ndarray: 出力系列, shapeは(N, T, inpu_size)

        Notation:
            N: バッチサイズ
            T: シーケンス長
        """
        xs = self.embed(xs)
        hs = xs
        for lstm in self.lstm_layers:
            hs = lstm(hs, h0) if h0 is not None else lstm(hs)
            h0 = None
        out = self.fc(hs)
        return out
