from abc import ABC, abstractmethod
from tensor import Tensor
import numpy as np
from operations.base import Operation

class Module(ABC):

    def __init__(self):
        self._parameters = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args,**kwargs)
    
    @abstractmethod
    def forward(self,*args,**kwargs):   
        pass
    
    def parameters(self):
        return self._parameters.values()
        
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()



class Linear(Module):
    def __init__(self,in_features: int,out_features: int, bias=True):
        super().__init__()
        self.weights = Tensor(np.random.rand(in_features,out_features))
        self._parameters["weights"] = self.weights
        if bias:
            self.bias = Tensor(np.zeros(out_features))
            self._parameters["bias"] = self.bias
        else:
            self.bias = None

    
    def forward(self, input: Tensor):
        matmul = (input @ self.weights)
        if self.bias:
            matmul += matmul + self.bias
        return matmul



class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

        for i,layer in enumerate(self.layers):
              for name, param in layer._parameters.items():
                    self._parameters[f"layer{i}_{name}"] = param # Prefix to avoid name clashes

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x


class AutoEncoder(Module):
    def __init__(self, input_dim: int, hidden_dim:int, activation: Operation = None):
        super().__init__()
        self.encoder = Linear(input_dim,hidden_dim)
        self.decoder = Linear(hidden_dim,input_dim)
        self.activation = activation


    def forward(self,x:Tensor):
        encoded = self.encoder(x)
        active = self.activation.apply(encoded)
        return self.decoder(active)
    

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


