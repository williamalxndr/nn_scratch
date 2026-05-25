import numpy as np

class Layer:
    def __init__(self, verbose=False):
        self.verbose=verbose

    def forward(self, x) -> np.ndarray: 
        """
        Returns the output of the layer

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
    
    def log(self, msg, force=False):
        if self.verbose:
            print(msg)
        elif force:
            print(msg)

    
class Optimizer:
    def __init__(self, lr, verbose=False):
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
    
    def log(self, msg, force=False):
        if self.verbose:
            print(msg)
        elif force:
            print(msg)