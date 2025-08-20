from pychu import as_tensor, as_array
from pychu.core import Tensor


def accuracy(y, t):
    y, t = as_tensor(y), as_tensor(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Tensor(as_array(acc))
