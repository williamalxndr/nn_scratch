import numpy as np
from base_layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        # TODO
        pass

    def forward(self, x):
        self.x = x
        return self._calculate_sigmoid()
    
    def backward(self, grad_out):
        grad = self._calculate_sigmoid() * (1 - self._calculate_sigmoid())
        return grad_out * grad

    def _calculate_sigmoid(self):
        return 1 / (1 + np.exp(-self.x))


class ReLU(Layer):
    def __init__(self):
        # TODO
        pass

    def forward(self, x):
        self.x = x
        return max(0, x)
    
    def backward(self, grad_out):
        grad = 1 if self.x > 0 else 0
        return grad_out * grad
    3


class Softmax(Layer):
    def __init__(self):
        # TODO
        pass

    def forward(self, x):
        sft = np.exp(x) / np.sum(np.exp(x))
        return sft
    
    def backward(self, grad_out):
        # TODO
        pass

