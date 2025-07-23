import numpy as np


def randn(shape, xp, dtype=np.float32):
    if hasattr(xp.random, "randn"):
        return xp.random.randn(*shape).astype(dtype)

    tmp = np.random.randn(*shape).astype(dtype)
    return xp.array(tmp)
