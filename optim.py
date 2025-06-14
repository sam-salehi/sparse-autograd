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



class ADAM: # checkout how adam works
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = params
        self.lr = lr
        self.beta1 = beta1  # First moment decay rate
        self.beta2 = beta2  # Second moment decay rate
        self.eps = eps      # Small constant for numerical stability
        
        # Initialize moment estimates
        self.m = []  # First moment (mean)
        self.v = []  # Second moment (variance)
        self.t = 0   # Time step
        
        # Initialize moment estimates for each parameter
        for param in self.parameters:
            self.m.append(np.zeros_like(param.data))
            self.v.append(np.zeros_like(param.data))
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            # Update first moment estimate (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update second moment estimate (RMSprop)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)