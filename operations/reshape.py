from tensor import Tensor
import numpy as np
from typing import Tuple
from .base import Operation

class Reshape(Operation):
    @staticmethod
    def apply(a: Tensor, *shape: int) -> Tensor:
        op = Reshape()
        op._prev = (a, shape)
        res = op._forward(a.data, shape)
        return Tensor(res,op)

    def _forward(self, a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return np.reshape(a, shape)

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        return np.reshape(grad_output, self._prev[0].data.shape) 