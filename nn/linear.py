import numpy as np
from .base_class import Layer, Optimizer
from .optimizer import *
from .loss import *

class Linear(Layer):
    def __init__(self, input_size, output_size, lr=1e-5, optimizer=GradientDescent):
        """
        Initialize a linear layer
        """
        super().__init__(False)

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr

        # Weights and bias
        self.w = np.random.rand(output_size, input_size)       # Weights shape = output_size x input_size
        self.b = np.random.rand(1, output_size)                # Bias shape = 1 x output_size

        # Weights and bias optimizer
        self.w_optimizer = optimizer()
        self.b_optimizer = optimizer()

        # Batch size training
        self.batch_size = None


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
        
        if x.shape[1] != self.input_size:
            raise ValueError("x's shape should be (batch_size, input_size)")

        if x.ndim == 1:
            x = x.reshape(-1, 1)
                    
        self.x = x  # This is used later for backward

        if self.batch_size is not None and self.x.shape[0] != self.batch_size:
            raise ValueError(f"Batch size must be {self.batch_size}")

        z = (x @ self.w.T) + self.b
        return z


    def backward(self, grad_out: np.ndarray) -> np.ndarray :
        """
        Perform a backward pass through the linear layer.

        Computes gradients with respect to weights, biases, and input,
        then updates weights and biases using the configured optimizer.

        Args:
            grad_out: Gradient of the loss w.r.t. this layer's output (dL/dz),
                    np.ndarray with shape (batch_size, output_size).

        Returns:
            dL/dx: Gradient of the loss w.r.t. the input, to be passed to
                the previous layer. 
                np.ndarray with shape: (batch_size, input_size)
        
        Modifies:
            self.dw: 
                dL/dw, np.ndarray with shape: (output_size, input_size)
            self.db: 
                dL/db, np.ndarray with shape: (1, output_size)

        Notes:
            Given z = Wx + b, gradients are derived as:

                dL/dW = dL/dz * x
                dL/db = dL/dz
                dL/dx = dL/dz * W  ← returned
        """
        self.log("Backwarding...")
        self.log("\n")

        # Argument validation
        if not isinstance(grad_out, np.ndarray):
            raise TypeError("grad_out should be np.ndarray")
        
        if self.batch_size is not None and grad_out.shape != (self.batch_size, self.output_size):
            err_msg = f"grad_out's shape is wrong. Should be ({self.batch_size, self.output_size}), instead ({grad_out.shape})"
            raise ValueError(err_msg)

        # Backwarding
        self.dw = grad_out.T @ self.x
        self.db = np.mean(grad_out, axis=0).reshape(-1, 1).T

        self._optimize()

        dx = grad_out @ self.dw

        self.log("Backwarding complete!")
        self.log("=================================")
        
        return dx
    
    def _optimize(self):
        """
        Optimize the weights and bias
        """
        self.log("Optimizing...")
        self.log("\n")

        self.w = self.w_optimizer.optimize(self.w, self.dw) # self.w shape: (output_size, input_size)
        self.b = self.b_optimizer.optimize(self.b, self.db) # self.b shape: (1, output_size)

        self.log("Optimizing complete!")
        self.log("=================================")


    def set_optimizer(self, optimizer):
        """
        Set the optimizer used for updating weights and biases.

        Args:
            optimizer: Optimizer class/instance (e.g. Adam, SGD) or a string
                    identifier (e.g. "adam", "sgd").

        Modifies:
            self.w_optimizer: Optimizer, set to be the desired optimizer
            self.b_optimizer: Optimizer, set to be the desired optimizer

        Example:

            # Using string
            layer = Linear(3, 5)
            layer.set_optimizer("adam")

            # Using optimizer class
            from optimizer import Adam
            layer = Linear(3, 5)
            layer.set_optimizer(Adam)
        """
        builder = OptimizerBuilder()
        if isinstance(optimizer, Optimizer):
            self.w_optimizer = optimizer()
            self.b_optimizer = optimizer()

        elif isinstance(optimizer, str):
            self.w_optimizer = builder.build(optimizer)
            self.b_optimizer = builder.build(optimizer)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size




if __name__ == "__main__":
    model_debug = Linear(3, 5)

    x = np.random.randn(1, 3)
    y_true = np.random.randn(1, 5)

    y = model_debug.forward(x)
    loss = mse(y, y_true)

    print(f"Loss: {loss}")
    model_debug.backward(loss.backward())




