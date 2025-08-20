import os
import sys

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu  # noqa
from pychu import Tensor  # noqa
import pychu.functions as F  # noqa

x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
indices = np.array([0, 0, 1])
batch = np.arange(len(indices))  # [0, 1, 2]
y = F.get_item(x, [1, 2])
print(y)
