import numpy as np

class Layer:
    def __init__(self, input_size, output_size, verbose=False):
        self.input_size = input_size
        self.output_size = output_size
        self.verbose=verbose

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
        raise NotImplementedError
    
    def backward(self, grad_out: np.ndarray) -> np.ndarray: 
        """
        Perform backward
        Args: 
            grad_out: The gradient of the loss w.r.t the output of this layer. dL/dy
                np.ndarray with shape (batch_size, output_size)
        Returns:
            grad: The gradient of the loss w.r.t the input of this layer (x). dL/dx
        """
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def log(self, msg, force=False):
        if self.verbose:
            print(msg)
        elif force:
            print(msg)

    
class Optimizer:
    def __init__(self, params, lr, verbose=False):
        """
        Initialize parameter
        Args:
            params: dict of theta and dtheta
        """
        self.params = params
        self.lr = lr
        self.verbose=verbose

    def step(self, theta: dict, dtheta: dict):
        """
        Returns the update for theta
        Args:
            theta: dict of np.ndarray
                    dictionary containing the thetas to be optimized.
                    Example:
                    {}
            dtheta: dict of np.ndarray
                    dictionary containing the dthetas corresponding with the theta.
                    Each must be the same shape as theta
        Returns:
            theta: np.ndarray with the same shape as the input
                    theta after optimized 
        
        Modify:
            theta
        
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