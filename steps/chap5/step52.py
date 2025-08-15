import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pychu  # noqa
import pychu.functions as F  # noqa
from pychu import DataLoader, datasets, optimizers  # noqa
from pychu.models import MLP  # noqa

max_epoch = 5
batch_size = 100

train_set = datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optim = optimizers.SGD().setup(model)


if pychu.gpu.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optim.update()
        sum_loss += float(loss.data) * len(t)  # type: ignore

    elapsed_time = time.time() - start
    print(f"epoch: {epoch + 1}, loss: {(sum_loss / len(train_set)):.4f}, "
          f"time: {elapsed_time:.4f}[sec]")
