import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.datasets  # noqa
import pychu  # noqa
from pychu.models import Model  # noqa
import pychu.functions as Func  # noqa
import pychu.layers as Layer  # noqa
import pychu.optimizers as optim  # noqa


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = Layer.RNN(hidden_size)
        self.fc = Layer.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


max_epoch = 100
hidden_size = 100
bptt_length = 30

train_set = pychu.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = optim.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += Func.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()  # type: ignore
            loss.unchain_backward()  # type: ignore
            optimizer.update()

    avg_loss = float(loss.data) / count  # type: ignore
    print(f"epoch {epoch + 1}, loss {avg_loss}")

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
