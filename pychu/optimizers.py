import math
import numpy as np
from pychu import gpu


class Optimizer:
    def __init__(self, lr=None):
        self.target = None
        self.hooks = []
        self.lr = lr

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # None以外のパラメータをリストにまとめる
        params = [p for p in self.target.params()  # type: ignore
                  if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            xp = gpu.get_array_module(param.data)
            param.data = gpu.to_xp(param, xp)
            param.grad = gpu.to_xp(param, xp)
            self.update_one(param)

    def update_one(self, param):
        # 子クラスでオーバライドされる想定
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

    def step(self):
        max_norm = 5.0
        total_norm = 0.0

        for param in self.target.params():  # type: ignore
            if param.grad is not None:
                total_norm += (param.grad ** 2).sum()
        total_norm = np.sqrt(total_norm)
        rate = max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in self.target.params():  # type: ignore
                if param.grad is not None:
                    param.grad *= rate

        for param in self.target.params():  # type: ignore
            if param.grad is not None:
                param.data -= self.lr * param.grad


# SGD関数(勾配降下法)
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        """
        W = W - lr * aL/aW
        勾配降下法の基本的な更新式
        """
        param.data -= self.lr * param.grad.data


# momentum関数
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        """
        W = W + v
        v = momentum * v - lr * aL/aW
        vは過去の勾配の移動平均
        """
        # idはアドレス(メモリ内での)を出力する
        v_key = id(param)
        if v_key not in self.vs:
            xp = gpu.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


# AdaGrad関数
class AdaGrad(Optimizer):
    """
    W = W - lr * aL/aW / sqrt(h + eps)
    hは過去の勾配の二乗和
    """
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = gpu.get_array_module(param.data)
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)


# Adam関数
class Adam(Optimizer):
    """
    W = W - lr * m / (sqrt(v) + eps)
    mは過去の勾配の移動平均
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def alpha(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.lr * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = gpu.get_array_module(param.data)

        # ★ grad を安全に取り出す（Tensorでも配列でもOK）
        grad = getattr(param.grad, "data", param.grad)

        # 以下は例: Adam の場合
        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
        if key not in self.vs:
            self.vs[key] = xp.zeros_like(param.data)

        m = self.ms[key] = self.beta1 * self.ms[key] + (1.0 - self.beta1) * grad
        v = self.vs[key] = self.beta2 * self.vs[key] + (1.0 - self.beta2) * (grad * grad)

        m_hat = m / (1.0 - self.beta1 ** self.t)
        v_hat = v / (1.0 - self.beta2 ** self.t)

        param.data = param.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
