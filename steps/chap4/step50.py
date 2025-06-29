import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pychu  # noqa
import pychu.datasets as datasets  # noqa
import pychu.functions as F  # noqa
from pychu import DataLoader, optimizers  # noqa
from pychu.models import MLP  # noqa

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = datasets.Spiral(train=True)
test_set = datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

train_losses = []
test_losses = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)  # type: ignore
        sum_acc += float(acc.data) * len(t)  # type: ignore
    print(f"epoch: {epoch + 1}")
    print(f"train loss: {sum_loss / len(train_set)},"
          f"accuracy: {(sum_acc / len(train_set)):.4f}")
    train_losses.append(sum_loss / len(train_set))

    sum_loss, sum_acc = 0, 0
    with pychu.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)  # type: ignore
            sum_acc += float(acc.data) * len(t)  # type: ignore

    print(f"test loss: {(sum_loss / len(test_set)):.4f}, "
          f"accuracy: {(sum_acc / len(test_set)):.4f}")
    test_losses.append(sum_loss / len(test_set))


plt.figure()
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
