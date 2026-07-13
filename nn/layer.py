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
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)   
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError()
    
    @abstractmethod
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

class Linear(Layer):
    def __init__(self, input_size, output_size, lr=1e-5, init="xavier", verbose=False):
        """
        Initialize a linear layer
        """
        super().__init__(verbose=verbose)
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr

        # Weights and bias
        if init == "xavier":
            self._xavier_init()
        elif init == "he":
            self._he_init()
        elif init == "random":
            self.w = np.random.rand(output_size, input_size)          # Weights shape = output_size x input_size
            self.b = np.random.randn(1, output_size)                  # Bias shape = 1 x output_size
        else:
            raise ValueError("unknown initializer")

        # Batch size training
        self.batch_size = None
    
    def parameters(self):
        return {
            id(self.w): self.w,
            id(self.b): self.b
        }
        
    def grads(self):
        return {
            id(self.w): self.dw,
            id(self.b): self.db
        }
    
    def forward(self, x: np.ndarray):
        """
        Perform the forward pass: z = Wx + b.


        Args:
            x: Input vector of shape (batch_size, input_size).

        Returns:
            z: Output vector of shape (batch_size, output_size).
        """

        # Argument validation
        if not isinstance(x, np.ndarray):
            raise TypeError("x's type should np.ndarray")
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        if x.shape[1] != self.input_size:
            raise ValueError("x's shape should be (batch_size, input_size)")
                    
        self.x = x  # This is used later for backward

        if self.batch_size is not None and self.x.shape[0] != self.batch_size:
            raise ValueError(f"Batch size must be {self.batch_size}")

        z = (x @ self.w.T) + self.b
        return z

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Argument validation
        if not isinstance(grad_out, np.ndarray):
            raise TypeError("grad_out should be np.ndarray")
        
        if self.batch_size is not None and grad_out.shape != (self.batch_size, self.output_size):
            err_msg = f"grad_out's shape is wrong. Should be ({self.batch_size, self.output_size}), instead ({grad_out.shape})"
            raise ValueError(err_msg)

        # Backwarding, computing gradient
        # dL/dw = dL/dz * dz/dw
        self.dw = (grad_out.T @ self.x) / self.x.shape[0]
        self.db = np.mean(grad_out, axis=0).reshape(-1, 1).T

        # Gradient clipping
        self.dw = self._clip_grad_norm(self.dw)
        self.db = self._clip_grad_norm(self.db)

        dx = grad_out @ self.w

        self.log("Backwarding complete!")
        self.log("=================================")
        
        return dx
    
    # ========= Helper methods =========
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def _xavier_init(self, dist="uniform"):
        if dist == "uniform":
            num = np.sqrt(6/(self.input_size+self.output_size))
            self.w = np.random.uniform(-num, num, (self.output_size, self.input_size))
            self.b = np.zeros((1, self.output_size))
        elif dist == "normal":
            num = np.sqrt(2/(self.input_size, self.output_size))
            self.w = np.random.normal(0, num, size=(self.output_size, self.input_size))
            self.b = np.zeros((1, self.output_size))

    def _he_init(self, dist="uniform"):
        var_w = np.sqrt(2/self.input_size)
        if dist == "uniform": 
            bound = np.sqrt(6/self.input_size)
            self.w = np.random.uniform(-bound, bound, (self.output_size, self.input_size))
        elif dist == "normal":
            self.w = np.random.normal(0, var_w, (self.output_size, self.input_size))
        self.b = np.zeros((1, self.output_size))

    def _clip_grad_norm(self, g:np.ndarray, max_norm=1):
        norm = np.sqrt(np.sum(np.square(g)))
        if norm > max_norm:
            g = g * (max_norm / norm)
        return g


class Network(Layer):
    def __init__(self, *layers: Layer, verbose=False):
        super().__init__(verbose=verbose)
        self.layers = []

        # Add each layer to network
        for layer in layers:
            self._add_layer(layer)
            
    def parameters(self):
        params = {}
        for layer in self.layers:
            params.update(layer.parameters())
        return params

    def grads(self):
        gts = {}
        for layer in self.layers:
            gts.update(layer.grads())
        return gts

    def forward(self, x: np.ndarray):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out

    def backward(self, grad_out: np.ndarray):
        for layer in reversed(self.layers):            
            grad_out = layer.backward(grad_out)
            
    def _add_layer(self, layer: Layer):
        """
        Add layer to the last position in network
        """
        if not isinstance(layer, Layer):
            raise ValueError("layer must be Layer object")
        
        self.layers.append(layer)


class ResidualBlock(Layer):
    ...