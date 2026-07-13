import numpy as np
from nn import Layer
from abc import ABC, abstractmethod

class ActivationFunction(Layer):        
    def parameters(self):
        return {}

    def grads(self):
        return {}

    @abstractmethod
    def forward(self, x: np.ndarray):
        """
        Args:
            x: the input data
                np.ndarray with shape (batch_size, size)
        Returns:
            sigmoid(x): np.ndarray 
        """
        raise NotImplementedError("Not implemeted in base class")
    
    @abstractmethod
    def backward(self, grad_out: np.ndarray):
        """
        Calculate the gradient of the loss w.r.t x
        Args: 
            grad_out: The gradient of the loss w.r.t the output, dL/dy
                        np.ndarray with shape (batch_size, size)
        
        Returns:
            grad: The gradient of the loss w.r.t the input, dL/dx
                    np.ndarray with shape (batch_size, size)
        """
        raise NotImplementedError("Not implemented in base class")


class Sigmoid(ActivationFunction):        
    def forward(self, x:np.ndarray):
        self.x = x
        return self._calculate_sigmoid()
    
    def backward(self, grad_out):
        grad = self._calculate_sigmoid() * (1 - self._calculate_sigmoid())
        return grad_out * grad

    def _calculate_sigmoid(self):
        return 1 / (1 + np.exp(-self.x))


class ReLU(ActivationFunction):
    def forward(self, x):
        self.x = x
        return np.maximum(np.zeros_like(x), x)
    
    def backward(self, grad_out):
        grad = np.where(self.x > 0, 1, 0)
        return grad_out * grad


class Softmax(ActivationFunction):
    def forward(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.s

    def backward(self, grad_out):
        s = self.s
        dot = np.sum(grad_out * s, axis=1, keepdims=True)
        dx = s * (grad_out - dot)
        return dx