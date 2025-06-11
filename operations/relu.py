from tensor import Tensor
import numpy as np
from .base import Operation

class ReLU(Operation):
    @staticmethod
    def apply(a: Tensor) -> Tensor:
        op = ReLU()
        op._prev = (a,)
        res = op._forward(a.data)
        return Tensor(res,op)

    def _forward(self, a: np.ndarray) -> np.ndarray:
        return np.maximum(0, a)

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self._prev[0].data > 0) 