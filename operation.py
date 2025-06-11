from tensor import Tensor
import numpy as np 
from typing import Tuple, Any, Union, Optional


# implementation motivated by https://github.com/minitorch/minitorch/tree/main/minitorch


# assuming everything requires gradient
class Operation:
    def __init__(self) -> None:
        self._prev = () # used to store previous matrix operations as required for backprop. Number of elements depends on operation

    @staticmethod
    def apply(input_tensors: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ 
        Used to perfrom forward pass in gragh
        """
        raise NotImplementedError

    def _forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        applies calculation. Stores value for backwards pass
        """
        raise NotImplementedError

    def _backward(self, grad_output: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Implements gradient proporgation
        """
        raise NotImplementedError



        




if __name__ == "__main__":
    n1 = np.array([1,2,3,4])
    n2 = np.array([2,4,3,2])
    n = 2
    t1 = Tensor(n1)
    t2 = Tensor(n2)

    t3 =  n * (t1 + t2)
    t4 = (n * t1 + n * t2)
