import numpy as np
from abc import ABC, abstractmethod

class Log(ABC):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, msg, force=False):
        if self.verbose:
            print(msg)
        elif force:
            print(msg)


class Layer(Log, ABC):
    def __init__(self, input_size, output_size, verbose=False):
        super().__init__(verbose)
        self.input_size = input_size
        self.output_size = output_size
        
    def parameters(self):
        raise NotImplementedError()
    
    def grads(self):
        raise NotImplementedError()

    def forward(self, x: np.ndarray) -> np.ndarray: 
        """
        Returns the output of the layer

        Args:
            x: the input data to be forwarded
                np.ndarray with shape (batch_size, input_size)
        Returns:
            y: the output data that gets forwarded
                np.ndarray with shape (batch_size, output_size)
        """
        raise NotImplementedError()
    
    def backward(self, grad_out: np.ndarray) -> np.ndarray: 
        """
        Perform backward
        Args: 
            grad_out: The gradient of the loss w.r.t the output of this layer. dL/dy
                np.ndarray with shape (batch_size, output_size)
        Returns:
            grad: The gradient of the loss w.r.t the input of this layer (x). dL/dx
        """
        raise NotImplementedError()
    
    def __call__(self, x):
        """
        Returns the output of the layer

        Args:
            x: the input data to be forwarded
                np.ndarray with shape (batch_size, input_size)
        Returns:
            y: the output data that gets forwarded
                np.ndarray with shape (batch_size, output_size)
        """
        return self.forward(x)
