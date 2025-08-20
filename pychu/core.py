import contextlib
import weakref

import numpy as np

import pychu

from typing import Any

"""
このファイルはTensorの演算子(+-*/%)などの設定
Tensor変数が使えるメソッド(T, shape, size)などの設定を行う
"""


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


# 関数をwith文で使えるようにするもの
# yieldの前
# try:
#    (     )   <- ここだwith文内の処理
#    yield
#    (     )   <- ここがwith文が終わった後の処理

@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


@contextlib.contextmanager
def no_grad():
    try:
        using_config("enable_backprop", False)
    finally:
        using_config("enable_backprop", True)


# 一時的に推論モードに変える
def test_mode():
    return using_config("train", False)


###############################################################################
# cupy
###############################################################################


array_types: tuple[type, ...] = (np.ndarray, )

try:
    import cupy as cp
    array_types += (cp.ndarray, )
except ImportError:
    pass

try:
    import mlx.core as mx
    array_types += (type(mx.array(0)), )
except ImportError:
    pass

# =============================================================================
# Tensor / Function
# =============================================================================


def as_tensor(obj: np.ndarray):
    """Tensorでないときに変換して返す

    Args:
        obj (Any): 入力

    Returns:
        Tensor: Tensorに返してから返す
    """
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


def as_array(x: int | float | np.ndarray, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Tensor:
    # これはnpなどのほかの変数と__array_priority__(標準では0.0)を比較して大きいほうの演算子を使う
    __array_priority__ = 1

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                print(f"{data} was {type(data)}. so changed to {array_types}")
                data = as_array(data, array_module=np)

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    def set_creator(self, func):
        self.creator = func

        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        """
        retain_grad: 途中の微分を覚えておくかどうか、デフォルトでは覚えない(False)
        create_graph: 2階微分以上を行うか行わないか、デフォルトでは行わない(False)
        """
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            # こうすることで今までndarrayで作られていたものではなくつながりを持った計算になる
            # つながりがあればそれに対してもまたそいつが何によって作られたのかなどがわかる
            xp = pychu.gpu.get_array_module(self.data)
            self.grad = Tensor(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            # gxsを作る前にusing_configを使うことでFunction内の
            # Config.back_propが変化しつながりを作るか作らないかを決めれる
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # これがx.grad += gx(インプレース)だとy.gradとx.gradが同じアドレスを参照してしまう
                        # つながりがなくなってしまう
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]  # type: ignore
        from pychu import functions
        return functions.reshape(self, shape)

    def transpose(self):
        from pychu import functions
        return functions.transpose(self)

    @property
    def T(self):
        from pychu import functions
        return functions.transpose(self)

    @property
    def shape(self):
        return getattr(self.data, "shape", ())

    @property
    def size(self):
        return getattr(self.data, "size", 1)

    @property
    def ndim(self):
        return getattr(self.data, "ndim", 0)

    @property
    def dtype(self):
        return getattr(self.data, "dtype", type(self.data))

    def __getitem__(self, slices):
        import pychu.functions as F
        return F.get_item(self, slices)

    def __len__(self):
        return getattr(self.data, "__len__", 1)

    def __repr__(self):
        """これはprint(x)などでTensorが呼ばれたときになんと返すかを決めれる処理"""
        if self.data is None:
            return "tensor(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "tensor(" + p + ")"

    def __neg__(self):
        return neg(self)

    def __pow__(self, c):
        return pow(self, c)

    def __floordiv__(self, other):
        return floordiv(self, other)

    def __div__(self, other):
        return div(self, other)

    def sum(self, axis=None, keepdims=False):
        import pychu.functions as F
        return F.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = pychu.gpu.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = pychu.gpu.as_gpu_array(self.data)

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()


class Function:
    # 変数と関数のつながりが作られる
    # __call__はFunctionを継承するクラスを使うとき
    # 例 f = Add()
    #    y = f(x0, x1) <-ここで呼ばれる
    def __call__(self, *inputs):
        inputs = [as_tensor(x) for x in inputs]  # type: ignore

        xs = [x.data for x in inputs]

        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Tensor(as_array(y)) for y in ys]

        # Configクラスのenable_backpropを呼び出しているだけ
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def backward(self, gy):
        # 子クラスでオーバライドされる
        raise NotImplementedError()


# =============================================================================
# 演算子クラスと関数
# =============================================================================


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            import pychu.functions as F
            gx0 = F.sum_to(gx0, self.x0_shape)
            gx1 = F.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1, pychu.gpu.get_array_module(x0.data))
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            import pychu.functions as F
            gx0 = F.sum_to(gx0, self.x0_shape)
            gx1 = F.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, pychu.gpu.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, pychu.gpu.get_array_module(x0.data))
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            import pychu.functions as F
            gx0 = F.sum_to(gx0, x0.shape)
            gx1 = F.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, pychu.gpu.get_array_module(x0.data))
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            import pychu.functions as F
            gx0 = F.sum_to(gx0, x0.shape)
            gx1 = F.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, pychu.gpu.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, pychu.gpu.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x0):
        return x0 ** self.c

    def backward(self, gy):
        x0 = self.inputs[0]
        return gy * (self.c * x0 ** (self.c - 1))


def pow(x, c):
    return Pow(c)(x)


class FloorDiv(Function):
    def forward(self, x0, x1):
        return x0 // x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * (1 // x1), gy * (-x0 // x1 ** 2)


def floordiv(x0, x1):
    return FloorDiv()(x0, x1)


class Mod(Function):
    def forward(self, x0, x1):
        return x0 % x1

    def backward(self, gy):
        x0, x1 = self.inputs
        q = x0 // x1
        return gy * 1, gy * (-q)


def mod(x, c):
    return Mod()(x, c)


def setup_tensor():
    Tensor.__add__ = add  # type: ignore
    Tensor.__radd__ = add  # type: ignore
    Tensor.__mul__ = mul  # type: ignore
    Tensor.__rmul__ = mul  # type: ignore
    Tensor.__neg__ = neg  # type: ignore
    Tensor.__sub__ = sub  # type: ignore
    Tensor.__rsub__ = rsub  # type: ignore
    Tensor.__truediv__ = div  # type: ignore
    Tensor.__rtruediv__ = rdiv  # type: ignore
    Tensor.__pow__ = pow  # type: ignore
    Tensor.__div__ = div  # type: ignore
    Tensor.__floordiv__ = floordiv  # type: ignore

    Tensor.matmul = pychu.functions.matmul  # type: ignore
    Tensor.dot = pychu.functions.matmul  # type: ignore


###############################################################################

class Parameter(Tensor):
    pass
