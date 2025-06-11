import numpy as np
from tensor import Tensor

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

if __name__ == "__main__":
    test_backward()


    

