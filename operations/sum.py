from tensor import Tensor
import numpy as np
from typing import Optional
from .base import Operation

class Sum(Operation):
    @staticmethod
    def apply(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        op = Sum()
        op._prev = (a, axis, keepdims)
        res = op._forward(a.data, axis, keepdims)
        return Tensor(res,op)

    def _forward(self, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._prev[1] is not None:  # if axis was specified
            shape = list(self._prev[0].data.shape)
            shape[self._prev[1]] = 1
            grad_output = np.reshape(grad_output, shape)
        return np.broadcast_to(grad_output, self._prev[0].data.shape) 