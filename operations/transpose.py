from tensor import Tensor
import numpy as np
from typing import Optional, Tuple
from .base import Operation

class Transpose(Operation):
    @staticmethod
    def apply(a: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
        op = Transpose()
        op._prev = (a, axes)
        res = op._forward(a.data, axes)
        return Tensor(res,op)

    def _forward(self, a: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        return np.transpose(a, axes)

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._prev[1] is None:  # if axes was None
            return np.transpose(grad_output)
        # If axes was specified, we need to transpose back
        axes = list(range(len(self._prev[1])))
        for i, j in enumerate(self._prev[1]):
            axes[j] = i
        return np.transpose(grad_output, axes) 
    