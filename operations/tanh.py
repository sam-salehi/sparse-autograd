import numpy as np 
from tensor import Tensor
from .base import Operation


class Tanh(Operation):
    @staticmethod
    def apply(a:Tensor) -> Tensor:
        op = Tanh()
        op._prev = (a,)
        res = op._forward(a.data)
        return Tensor(res,op)
    
    def _forward(self,a:np.ndarray) -> np.ndarray:
        x = np.clip(a, -50, 50)  # Prevent extreme values
        return np.tanh(x)
    
    def _backward(self,grad_output: np.ndarray) -> np.ndarray:
        s = self._forward(self._prev[0].data)
        return grad_output * (1-s**2)
