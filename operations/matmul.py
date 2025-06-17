from tensor import Tensor
import numpy as np
from typing import Tuple
from .base import Operation

class MatMul(Operation):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        op = MatMul()
        op._prev = (a, b)
        res = op._forward(a.data, b.data)
        return Tensor(res, op)

    def _forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self._prev
        # allowing 2d shape for batch support. 
        
        if b.data.ndim == 1:
            # a: (m,n)
            # b: (n,)
            grad_a = np.outer(grad_output, b.data)  # (m,n)
            grad_b = a.data.T @ grad_output  # (n,)
        else:
            grad_output_2d = np.atleast_2d(grad_output)
            a_data_2d = np.atleast_2d(a.data)
            b_data_2d = np.atleast_2d(b.data)
            grad_a = grad_output_2d @ b_data_2d.T  # (batch_size, in_features)
            grad_b = a_data_2d.T @ grad_output_2d  # (in_features, out_features)
        
        if a.data.ndim == 1:
            grad_a = grad_a.squeeze(axis=0)
        
        return grad_a, grad_b
    



# a: [265,12]
# b: 