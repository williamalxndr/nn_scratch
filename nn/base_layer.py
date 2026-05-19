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

    