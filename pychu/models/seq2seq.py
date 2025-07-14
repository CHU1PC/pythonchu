import pychu.functions as F
import pychu.layers as L

from pychu.models import Model


class seq2seq(Model):
    def __init__(self, input_size, embedding_dim, hidden_size,
                 num_layers=1, padding_idx=None):
        super().__init__()

        self.encoder = L.Encoder(input_size, embedding_dim, hidden_size,
                                 num_layers, padding_idx)

        self.decoder = L.Decoder(input_size, embedding_dim, hidden_size,
                                 num_layers, padding_idx)

    def forward(self, xs, ts):
        """

        Args:
            xs (ndarray or Variable): 入力系列(時系列データ)
            ts (ndarray or Variabel): ターゲット系列(時系列データ), xsに対する正解データ
        """
        h = self.encoder.last_hidden(xs)
        score = self.decoder(ts, h)
        loss = F.time_softmax_cross_entropy(score, ts)
        return loss
