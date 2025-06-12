import numpy as np

#  Gradient Descent implementation
class GD: 
    def __init__(self,params, lr=0.01):
        self.parameters = params
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                # Sum or average over the batch dimension
                grad = np.mean(param.grad, axis=0) 
                param.data -= self.lr * grad
                


# Sochastic Gradient Descent Implementation 
# should just be implemented in training pass.
class SGD:
    def __init__(self, params, lr=0.01):
        self.parameters = params
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                # expectig data not to be batched
                # Update using the gradient (no averaging needed for SGD)
                param.data -= self.lr * param.grad 