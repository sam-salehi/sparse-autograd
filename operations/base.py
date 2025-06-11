from abc import ABC, abstractmethod
from tensor import Tensor
import numpy as np
from typing import Tuple, Any, Union, Optional

class Operation(ABC):
    def __init__(self) -> None:
        self._prev = () # used to store previous matrix operations as required for backprop. Number of elements depends on operation

    @staticmethod
    @abstractmethod
    def apply(input_tensors: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ 
        Used to perform forward pass in graph
        """
        pass

    @abstractmethod
    def _forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Applies calculation. Stores value for backwards pass
        """
        pass

    @abstractmethod
    def _backward(self, grad_output: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Implements gradient propagation
        """
        pass 