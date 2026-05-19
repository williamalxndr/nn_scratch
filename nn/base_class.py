import numpy as np

class Layer:
    def __init__(self, verbose=True):
        self.verbose=verbose

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
    
    def log(self, msg):
        if self.verbose:
            print(msg)

    
class Optimizer:
    def __init__(self, lr, verbose=True):
        self.lr = lr

        self.verbose=verbose

    def optimize(self, theta, dtheta):
        """
        Returns the update for theta
        For example,
        SGD:
        theta_t+1 <-- theta_t - lr * dtheta
        returns theta_t+1
        """
        raise NotImplementedError
    
    def log(self, msg):
        if self.verbose:
            print(msg)