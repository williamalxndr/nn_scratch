import numpy as np
from base_class import Layer, Optimizer
from optimizer import *
from loss import *

class Linear(Layer):
    def __init__(self, input_size, output_size, lr=0.01, optimizer=GradientDescent, verbose=False):
        """
        Initialize a linear layer
        """
        super().__init__(False)

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr

        # Weights and bias
        self.w = np.random.rand(output_size, input_size)
        self.b = np.random.rand(output_size)

        # Weights and bias optimizer
        self.w_optimizer = optimizer()
        self.b_optimizer = optimizer()


    def forward(self, x: np.ndarray):
        """
        Perform the forward pass: z = Wx + b.

        Args:
            input: Input vector of shape (input_size,) or (batch_size, input_size).

        Returns:
            Output vector of shape (output_size,) or (batch_size, output_size).
        """
        self.log("Forwarding...")
        self.log("\n")


        self.x = x  # This is used later for backward

        self.log(f"w: {self.w}")
        self.log(f"x: {x}")
        self.log(f"b: {self.b}")
        
        b = self.b
        if x.shape[1] > 1:
            b = np.repeat(b.reshape(-1, 1), x.shape[1], axis=1)

        z = self.w @ x + b

        # self.log(f"z: {z}")
        self.log(f"z.shape: {z.shape}")

        self.log("Forwarding complete!")
        self.log("=================================")

        return z


    def backward(self, grad_out) -> np.ndarray :
        """
        Perform a backward pass through the linear layer.

        Computes gradients with respect to weights, biases, and input,
        then updates weights and biases using the configured optimizer.

        Args:
            grad_out: Gradient of the loss w.r.t. this layer's output (dL/dz),
                    shape (output_size,) or (batch_size, output_size).

        Returns:
            dL/dx: Gradient of the loss w.r.t. the input, to be passed to
                the previous layer. Shape matches the input x.

        Notes:
            Given z = Wx + b, gradients are derived as:

                dL/dW = dL/dz * x
                dL/db = dL/dz
                dL/dx = dL/dz * W  ← returned
        """
        self.log("Backwarding...")
        self.log("\n")

        self.dw = grad_out @ self.x.T
        self.db = np.mean(grad_out, axis=1)

        self._optimize()

        dx = self.w.T @ grad_out

        self.log("Backwarding complete!")
        self.log("=================================")
        return dx
    
    def _optimize(self):
        """
        Optimize the weights and bias
        """
        self.log("Optimizing...")
        self.log("\n")

        self.log(f"weights before: {self.w}")
        self.log(f"bias before: {self.b}")

        self.w = self.w_optimizer.optimize(self.w, self.dw)
        self.b = self.b_optimizer.optimize(self.b, self.db)

        self.log(f"weights after: {self.w}")
        self.log(f"bias after: {self.b}")

        self.log("Optimizing complete!")
        self.log("=================================")


    def set_optimizer(self, optimizer):
        """
        Set the optimizer used for updating weights and biases.

        Args:
            optimizer: Optimizer class/instance (e.g. Adam, SGD) or a string
                    identifier (e.g. "adam", "sgd").

        Example::

            # Using string
            layer = Linear(3, 5)
            layer.set_optimizer("adam")

            # Using optimizer class
            from optimizer import Adam
            layer = Linear(3, 5)
            layer.set_optimizer(Adam)
        """
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer

        elif isinstance(optimizer, str):
            self.optimizer = OptimizerBuilder.build(optimizer)



if __name__ == "__main__":
    model_debug = Linear(3, 5)

    x = np.random.randn(3, 2)
    y_true = np.random.randn(5, 2)

    y = model_debug.forward(x)
    loss = mse(y, y_true)

    print(f"Loss: {loss}")
    model_debug.backward(loss.backward())




