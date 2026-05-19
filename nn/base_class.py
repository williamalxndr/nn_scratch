import numpy as np

class Layer:
    def __init__(self):
        pass

    def forward(self, x) -> np.ndarray: 
        """
        Returns layer(x)

        Example usage:
        from linear import Linear
        net = Linear(3, 5)
        x = np.random.randn(3)

        net.forward(x) 
        """
        raise NotImplementedError
    
    def backward(self, x) -> np.ndarray: 
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)

    
class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def optimize(self, dtheta):
        """
        Returns the update for theta
        For example,
        SGD:
        theta_t+1 <-- theta_t - lr * dtheta
        returns theta_t+1
        """
        raise NotImplementedError