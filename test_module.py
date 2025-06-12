import numpy as np
from module import Module, Linear, Sequential
from tensor import Tensor

def test_linear_layer():
    # Test Linear layer initialization and forward pass
    in_features = 3
    out_features = 2
    linear = Linear(in_features, out_features)
    
    # Create a test input
    input_data = Tensor(np.array([[1.0, 2.0, 3.0]]))
    
    # Forward pass
    output = linear(input_data)
    
    # Check output shape
    assert output.shape == (1, out_features), f"Expected shape (1, {out_features}), got {output.shape}"

    print(input_data.grad)
    output.backward()
    print(input_data.grad)
    
    # Check if parameters exist
    assert "bias" in linear._parameters, "Bias parameter not found"
    assert linear.weights is not None, "Weights not initialized"

# def test_sequential():
#     # Test Sequential module with multiple layers
#     model = Sequential(
#         Linear(3, 4),
#         Linear(4, 2)
#     )
    
#     # Create a test input
#     input_data = Tensor(np.array([[1.0, 2.0, 3.0]]))
    
#     # Forward pass
#     output = model(input_data)
    
#     # Check output shape
#     assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    
#     # Check if parameters are properly registered
#     assert len(model._parameters) == 4, f"Expected 4 parameters (2 layers * 2 params), got {len(model._parameters)}"
    
#     # Test zero_grad
#     model.zero_grad()
#     for param in model.parameters():
#         assert param.grad is None or np.all(param.grad == 0), "Gradients not properly zeroed"

# now make loss function.

if __name__ == "__main__":
    print("Running tests...")
    test_linear_layer()
    # test_sequential()
    print("All tests passed!") 