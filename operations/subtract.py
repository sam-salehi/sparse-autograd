from tensor import Tensor
import numpy as np
from typing import Tuple
from .base import Operation

class Subtract(Operation):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        op = Subtract()
        op._prev = (a, b)
        res = op._forward(a.data, b.data)
        return Tensor(res, op)

    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output 