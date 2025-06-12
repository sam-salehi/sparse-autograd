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
        # Ensure 2D shapes for batch support
        grad_output_2d = np.atleast_2d(grad_output)
        a_data_2d = np.atleast_2d(a.data)
        
        # Gradient w.r.t. input: grad_output @ W.T
        grad_a = grad_output_2d @ b.data.T  # (batch_size, in_features)
        # Gradient w.r.t. weights: a.T @ grad_output
        grad_b = a_data_2d.T @ grad_output_2d  # (in_features, out_features)
        
        # Squeeze if original input was 1D
        if a.data.ndim == 1:
            grad_a = grad_a.squeeze(axis=0)
        # Do not squeeze grad_b, as it is always 2D

        
        return grad_a, grad_b