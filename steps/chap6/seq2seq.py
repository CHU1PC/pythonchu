import os
import sys

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pychu.models import seq2seq  # noqa
import pychu


# ハイパーパラメータ例
vocab_size = 1000
embedding_dim = 128
hidden_size = 256
num_layers = 2
padding_idx = 0

# モデルのインスタンス化
model = seq2seq(vocab_size, embedding_dim,
                hidden_size, num_layers, padding_idx)

# ダミーデータ（実際はデータローダ等で用意）
N, T = 32, 20  # バッチサイズ, シーケンス長
xs = np.random.randint(1, vocab_size, (N, T))  # 入力系列
ts = np.random.randint(1, vocab_size, (N, T))  # ターゲット系列

# 学習例（順伝播と損失計算）
loss = model(xs, ts)
print("loss:", loss)
