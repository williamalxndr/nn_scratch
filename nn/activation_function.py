import numpy as np
from nn.base_class import Layer

class ActivationFunction(Layer):
    def __init__(self, size):
        super().__init__(size, size)

    def forward(self, x: np.ndarray):
        """
        Args:
            x: the input data
                np.ndarray with shape (batch_size, size)
        Returns:
            sigmoid(x): np.ndarray 
        """
        raise NotImplementedError
    
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
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, x:np.ndarray):
        """
        Args:
            x: the input data
                np.ndarray with shape (batch_size, size)
        Returns:
            sigmoid(x): np.ndarray 
        """
        self.x = x
        return self._calculate_sigmoid()
    
    def backward(self, grad_out):
        """
        Calculate the gradient of the loss w.r.t x
        Args: 
            grad_out: The gradient of the loss w.r.t the output, dL/dy
                        np.ndarray with shape (batch_size, size)
        
        Returns:
            grad: The gradient of the loss w.r.t the input, dL/dx
                    np.ndarray with shape (batch_size, size)
        """
        grad = self._calculate_sigmoid() * (1 - self._calculate_sigmoid())
        return grad_out * grad

    def _calculate_sigmoid(self):
        return 1 / (1 + np.exp(-self.x))


class ReLU(ActivationFunction):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, x):
        self.x = x
        return np.maximum(np.zeros_like(x), x)
    
    def backward(self, grad_out):
        grad = np.where(self.x > 0, 1, 0)
        return grad_out * grad


class Softmax(ActivationFunction):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, x):
        sft = np.exp(x) / np.sum(np.exp(x))
        return sft
    
    def backward(self, grad_out):
        # TODO
        pass
        
class ActivationFunctionBuilder:
    def __init__(self):
        pass

    def build(self, type, size) -> ActivationFunction:
        if type.lower() == "sigmoid":
            return Sigmoid(size)
        elif type.lower() == "relu":
            return ReLU(size)
        elif type.lower() == "softmax":
            return Softmax(size)
        else:
            raise ValueError(f"{type} has not been implemented yet")
        