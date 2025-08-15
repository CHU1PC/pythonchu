import os
import weakref
import numpy as np
from pychu import gpu
from pychu.core import Parameter


class Layer:
    def __init__(self):
        """
        self._paramsにはLayerのインスタンスが持つParameter, Layerクラスを持つ
        """
        self._params = set()
        self.training = True

    def __setattr__(self, name, value):
        """インスタンス変数を宣言する際に呼び出されるメソッド

        Args:
            name (str): 名前を保持するための変数
            value (int, float, double)): 値を保持するための変数
        """
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs):
        raise NotImplementedError

    def params(self):
        """
        もしobjがLayerならLayerのなかのパラメータを取り出し
        もしobjがParameterならばそのまま返す
        """
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                # obj.params()のなかのparamsをyieldで取り出す
                yield from obj.params()
            else:
                yield obj

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            # objがLayerであればparent_keyは最初にnameそこからname/nameとどんどん増えていく
            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):
                # objがLayerなら再帰でLayerの中身を調べる
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def train(self):
        """学習モードに変える. defaultでは学習モード
        すべてのParameterをtraining = Falseに変える"""
        self.training = True
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                obj.train()

    def eval(self):
        """推論モードに変える
        すべてのParameterをtraining = Trueに変える"""
        self.training = False
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                obj.eval()

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def save_weights(self, path):
        # GPU上にある時cupyを利用しているため, numpyを使うためにCPUへ
        self.to_cpu()

        params_dict: dict = {}
        # params_dictに_flatten_paramsでLayerの構成を格納する
        self._flatten_params(params_dict)

        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}

        # 途中でエラーが発生したときに未完成のファイルの残さないためにremoveする
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict: dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# dropout層
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout_ratio = p

    def forward(self, x):
        if self.training:
            xp = gpu.get_array_module(x)
            try:
                mask = xp.random.rand(*x.shape) > self.dropout_ratio
            except Exception:
                mask = xp.random.uniform(0.0, 1.0, shape=x.shape)
            scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y = x * mask / scale
            return y
        else:
            return x


class TimeDropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout_ratio = p

    def forward(self, xs):
        if self.training:
            xp = gpu.get_array_module(xs)
            N, T, D = xs.shape
            # 各時系列ごとに同じmaskを使う
            try:
                mask = xp.random.rand(N, 1, D) > self.dropout_ratio
            except Exception:
                mask = xp.random.uniform(0.0, 1.0, shape=(N, 1, D))
            scale = xp.array(1.0 - self.dropout_ratio).astype(xs.dtype)
            y = xs * mask / scale
            return y
        else:
            return xs
