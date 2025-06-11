from tensor import Tensor
import numpy as np
from .base import Operation

class Sigmoid(Operation):
    @staticmethod
    def apply(a: Tensor) -> Tensor:
        op = Sigmoid()
        op._prev = (a,)
        res = op._forward(a.data)
        return Tensor(res,op)

    def _forward(self, a: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-a))

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        s = self._forward(self._prev[0].data)
        return grad_output * s * (1 - s) 