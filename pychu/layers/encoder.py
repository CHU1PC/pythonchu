# Encoder layer
import numpy as np
from pychu.layers import Layer, TimeEmbedding, TimeLSTM


class Encoder(Layer):
    def __init__(self, input_size, embedding_dim, hidden_size,
                 num_layers=1, padding_idx=None):
        """Encoderで使われるレイヤーを初期化する

        Args:
            input_size (int): 入力言語の語句数(vocab_size)
            embedding_dim (int): 埋め込みベクトルの次元数
            hidden_size (int): LSTMレイヤーの隠れ層の次元数
            num_layers (int): LSTMレイヤーの層数. Defaults to 1.
            padding_idx (int): paddingのindexを与える. Defaults to None
        """
        super().__init__()
        self.padding_idx = padding_idx

        # Embedding層
        embed_W = np.random.randn(input_size,
                                  embedding_dim).astype(np.float32) * 0.01
        # 単語をテンソル化
        self.embed = TimeEmbedding(embed_W)

        # LSTM層
        self.lstm_layers = []
        for i in range(num_layers):
            in_size = embedding_dim if i == 0 else hidden_size
            lstm = TimeLSTM(hidden_size, in_size)
            setattr(self, f"lstm_{i}", lstm)
            self.lstm_layers.append(lstm)

    def reset_state(self):
        for lstm in self.lstm_layers:
            lstm.reset_state()

    def last_hidden(self, xs):
        hs = self.forward(xs)
        return hs[:, -1, :]

    def forward(self, xs):
        mask = None
        if self.padding_idx is not None:
            mask = (xs != self.padding_idx)

        # xs: (N, T)入力系列 -> それぞれの単語を(N, T, D), バッチサイズ, シーケンス長, Dは埋め込みベクトルの次元数
        xs = self.embed(xs)

        if mask is not None:
            xs = xs * mask[:, :, None]

        hs = xs
        for lstm in self.lstm_layers:
            hs = lstm(hs)

        return hs
