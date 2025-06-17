from tensor import Tensor
from operations import matmul
import numpy as np


# [2,3,4]
# [1,2,3]

a = Tensor(np.random.rand(2,8))
b = Tensor(np.random.rand(8,1))

c = a @ b
print(c)
c.backward()
print(a.grad)
print(b.grad)