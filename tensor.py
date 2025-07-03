import numpy as np
from typing import Union, Optional, Tuple, Any, List, Set


class Tensor:
    data: np.ndarray
    grad: Optional[np.ndarray]

    _op: Optional[Any]  # Operation that created this tensor 
    
    def __init__(self, data: np.ndarray,op: Optional[Any] = None) -> None:
        self.data = data
        self.grad = None
        self._op = op 

    def __str__(self) -> str:
        return str(self.data)



    def _build_topo(self) -> List['Tensor']: 
        """Build topological ordering of tensors in the computation graph. (DFS topological sorting)"""
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()
        
        def build(tensor: 'Tensor') -> None:
            if tensor not in visited:
                visited.add(tensor)
                if tensor._op is not None:
                    for prev_tensor in tensor._op._prev:
                        build(prev_tensor)
                topo.append(tensor)
        
        build(self)
        return topo

    def backward(self) -> None:
        """Compute gradients for all tensors in the computation graph using topological sorting."""
        # Initialize gradient of the output tensor
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # Get topological ordering of tensors
        topo = self._build_topo()
        
        # Compute gradients in reverse topological order
        for tensor in reversed(topo):
            if tensor._op is not None:
                grads = tensor._op._backward(tensor.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                
                for prev_tensor, grad in zip(tensor._op._prev, grads):
                    if prev_tensor.grad is None:
                        prev_tensor.grad = grad
                    else:
                        prev_tensor.grad += grad

    def zero_grad(self):
        """Reset gradients"""
        self.grad = np.zeros_like(self.data)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        from operations.add import Add
        return Add.apply(self, other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        from operations.subtract import Subtract
        return Subtract.apply(self, other)

    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        from operations.multiply import Multiply
        if isinstance(other, (int, float)):
            # Convert scalar to tensor with same shape as self
            other = Tensor(np.full_like(self.data, other))
        return Multiply.apply(self, other)

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        from operations.divide import Divide
        return Divide.apply(self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':  # @
        from operations.matmul import MatMul
        return MatMul.apply(self, other)

    def __radd__(self, other: Union[int, float]) -> 'Tensor':
        from operations.add import Add
        return Add.apply(self, other)

    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        from operations.multiply import Multiply
        return Multiply.apply(self, other)

    def __rsub__(self, other: Union[int, float]) -> 'Tensor':
        from operations.subtract import Subtract
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return Subtract.apply(other, self)

    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        from operations.divide import Divide
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return Divide.apply(other, self)

    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        from operations.matmul import MatMul
        return MatMul.apply(self, other)

    def relu(self) -> 'Tensor':
        from operations.relu import ReLU
        return ReLU.apply(self)

    def sigmoid(self) -> 'Tensor':  
        from operations.sigmoid import Sigmoid
        return Sigmoid.apply(self)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        from operations.sum import Sum
        return Sum.apply(self, axis, keepdims)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor': 
        from operations.mean import Mean
        return Mean.apply(self, axis, keepdims)

    def reshape(self, *shape: int) -> 'Tensor':
        from operations.reshape import Reshape
        return Reshape.apply(self, *shape)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        from operations.transpose import Transpose
        return Transpose.apply(self, axes)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
