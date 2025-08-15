import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu  # noqa
from pychu.models import Model  # noqa
from pychu import SeqDataLoader  # noqa
import pychu.functions as F  # noqa
import pychu.layers as L  # noqa
import pychu.optimizers as optim  # noqa
import pychu.datasets  # noqa

max_epoch = 100
batch_size = 64
hidden_size = 10
bptt_length = 30

train_set = pychu.datasets.SinCurve(train=True)

dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


model = BetterRNN(hidden_size, 1)
optimizer = optim.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss_sum, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss = F.mean_squared_error(y, t)
        loss_sum += float(loss.data)  # type: ignore
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()  # type: ignore
            loss.unchain_backward()  # type: ignore
            optimizer.update()
    avg_loss = float(loss_sum) / count  # type: ignore
    print(f"| epoch {epoch + 1}| loss {avg_loss}")

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with pychu.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
