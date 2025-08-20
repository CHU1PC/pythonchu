import pychu
from pychu import as_tensor
from pychu import gpu


def dropout(x, dropout_ratio=0.1):
    x = as_tensor(x)

    if pychu.Config.train:
        xp = gpu.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
