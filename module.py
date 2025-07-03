from abc import ABC, abstractmethod
from tensor import Tensor
import numpy as np

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
    
    def named_parameters(self):
        return self._parameters
        
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()



class Linear(Module):
    def __init__(self,in_features: int,out_features: int, bias=True):
        super().__init__()
        scale = np.sqrt(2.0 / (in_features + out_features)) # Xavier initalization
        self.weights = Tensor(np.random.randn(in_features, out_features) * scale)
        self._parameters["weight"] = self.weights
        if bias:
            self.bias = Tensor(np.zeros(out_features))
            self._parameters["bias"] = self.bias
        else:
            self.bias = None

    
    def forward(self, input: Tensor):
        matmul = (input @ self.weights)
        if self.bias:
            matmul = matmul + self.bias
        return matmul


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

        for i,layer in enumerate(self.layers):
              for name, param in layer._parameters.items():
                    self._parameters[f"layer{i}_{name}"] = param # Prefix to avoid name clashes

    def forward(self,x):
        for _,layer in enumerate(self.layers):
            x = layer(x)
        return x


class AutoEncoder(Module):
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()
        self.encoder = Linear(input_dim,hidden_dim,True)
        self.decoder = Linear(hidden_dim,input_dim,True)


    def forward(self,x:Tensor):
        return self.decoder(self.encoder(x))
    

    def named_parameters(self):
        return self.encoder.named_parameters() | self.decoder.named_parameters()

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())



class BigAutoEncoder(Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,z_size):
        super().__init__()
        self.fe1 = Linear(input_size,hidden_size1)
        self.fe2 = Linear(hidden_size1,hidden_size2)
        self.fe3 = Linear(hidden_size2,z_size)

        self.encoder = Sequential(self.fe1,self.fe2,self.fe3)

        self.fd1 = Linear(z_size,hidden_size2)
        self.fd2 = Linear(hidden_size2,hidden_size1)
        self.fd3 = Linear(hidden_size1,input_size)

        self.decoder = Sequential(self.fd1,self.fd2,self.fd3)

    def forward(self,x:Tensor):
        raise NotImplementedError
    
    def named_parameters(self):
        return self.encoder.named_parameters() | self.decoder.named_parameters()
    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())




