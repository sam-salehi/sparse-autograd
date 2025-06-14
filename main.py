import numpy as np
from tensor import Tensor
from operations.loss import MSELoss
from operations.sum import Sum
from module import AutoEncoder

def test_backward():
    # Create input tensors
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array([1.0, 2.0]))
    
    # Create a computation graph: z = (x + y) * (x - y)
    z = (x + y) * (x - y)
    
    # Compute gradients
    z.backward()
    
    # Expected gradients:
    # dz/dx = 2x (from (x+y)*(x-y) = x^2 - y^2)
    # dz/dy = -2y
    expected_dx = np.array([4.0, 6.0])  # 2x
    expected_dy = np.array([-2.0, -4.0])  # -2y
    
    print("Input x:", x.data)
    print("Input y:", y.data)
    print("Output z:", z.data)
    print("\nGradients:")
    print("dz/dx:", x.grad)
    print("dz/dy:", y.grad)
    print("\nExpected gradients:")
    print("Expected dz/dx:", expected_dx)
    print("Expected dz/dy:", expected_dy)
    
    # Verify gradients
    assert np.allclose(x.grad, expected_dx), "Gradient for x is incorrect"
    assert np.allclose(y.grad, expected_dy), "Gradient for y is incorrect"
    print("\nAll gradient tests passed!")

def loss():
    actual = Tensor(np.array([1.0, 2.0, 3.0]))  # shape: (3,)
    pred = Tensor(np.array([1.1, 1.9, 3.1]))    # shape: (3,)
    loss = MSELoss.apply(actual, pred)          # scalar output
    print(loss)

    # Batch of samples
    actual = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))  # shape: (2, 2)
    pred = Tensor(np.array([[1.1, 1.9], [2.9, 4.1]]))    # shape: (2, 2)
    loss = MSELoss.apply(actual, pred)                   # scalar output
    print(loss)


def matt():
    model = AutoEncoder(5,6)
    k = Tensor(np.random.rand(5))
    res = model(k)
    loss = MSELoss.apply(k,res)
    print(loss)
    print("Gradients")
    print(k.grad)
    loss.backward()
    print(res.grad)
    print(k.grad)
    params = list(model.parameters())  
    print([x.data for x in params])


# play around with gradient here. Try to get whats wrong.
# make a sample linear model to play around with as well.
if __name__ == "__main__":  
    x = Tensor(np.array([3.0,5]))
    z = Tensor(np.array([3,3]))
    y = x * z
    l = Sum.apply(y)
    print(l)
    l.backward()
    print(x.grad)   # Expect 6.0



    

