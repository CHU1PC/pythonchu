import numpy as np

xp_gpu = np
gpu_backend = "cpu"
gpu_enable = False
try:
    import cupy as cp
    xp_gpu = cp
    gpu_backend = "cupy"
    gpu_enable = True
except ImportError:
    try:
        import mlx.core as mx
        xp_gpu = mx
        gpu_backend = "metal"
        gpu_enable = True
    except ImportError:
        pass

from pychu import Variable  # noqa

print("-" * 50, "\n", gpu_backend, "\n", "-" * 50)


def get_array_module(x):
    """xがnumpyかcupyかを返す

    Args:
        x (Variable, ndarray(cupy or numpy)): input

    Returns:
        xp (module): numpy or cupy module
    """
    if isinstance(x, Variable):
        x = x.data

    if gpu_backend == "cuda":
        xp = cp.get_array_module(x)
        return xp
    elif gpu_backend == "metal":
        return mx
    else:
        return np


def as_numpy(x):
    """Numpy配列に変換する

    Args:
        x (Variable, ndarray(cupy or numpy)): input

    Returns:
        np.ndarray: Numpy配列
    """
    if isinstance(x, Variable):
        x = x.data

    if gpu_backend == "cuda":
        return cp.asnumpy(x)
    elif gpu_backend == "metal":
        return x.to_numpy()
    return np.asarray(x)


def as_gpu_array(x):
    """任意の配列をGPU(Cupy / MLX)対応配列に変換する

    Args:
        x (Variable, ndarray(cupy or numpy)): input

    Raises:
        Exception: CuPyがインストールされていない場合

    Returns:
        cupy.ndarray: CuPy配列
    """
    if isinstance(x, Variable):
        x = x.data

    return xp_gpu.asarray(x)


def canonical_dtype(dtype, xp):
    if gpu_backend == "metal":
        if isinstance(dtype, (np.dtype, cp.dtype if gpu_backend == "cuda"
                              else np.dtype)):
            dtype = str(dtype)
        if isinstance(dtype, str):
            dtype = getattr(xp, dtype)
    return dtype


def to_xp(arr, xp):
    if isinstance(arr, Variable):
        arr = arr.data
    if xp is np:
        return np.asarray(arr)

    elif hasattr(xp, "asarray"):
        return xp.asarrayn(arr)

    elif hasattr(xp, "array"):
        return xp.array(arr)

    else:
        return arr
