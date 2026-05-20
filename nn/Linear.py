import numpy as np
from nn.base_class import Layer, Optimizer
from optimizer import *

class Linear(Layer):
    def __init__(self, input_size, output_size, lr=0.01, optimizer=GradientDescent, verbose=False):
        """
        Initialize a linear layer
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.verbose = verbose

        # Weights and bias
        self.w = np.random.rand(output_size, input_size)
        self.b = np.random.rand(output_size)

        # Weights and bias optimizer
        self.w_optimizer = optimizer()
        self.b_optimizer = optimizer()


        self.log(f"w: {self.w}")
        self.log(f"b: {self.b}")


    def forward(self, x):
        """
        --- Input ---
        input: vector with (input_size, 1) dimension

        --- Output ---
        Return: w * input + b
        """
        self.x = x  # This is used later for backward

        # z = wx + b
        z = self.w @ x + self.b
        self.log(f"z: {z}")
        return z


    def backward(self, grad_out) -> np.ndarray :
        """
        --- Input ---
        grad_out: gradient from the last layer


        --- Output ---
        Update self.weights and self.bias
        Return: dL/dx

        This is used for backpropagation
        Outputs the gradient for backward passing the next layer

        
        Explanation
        last layer's variable is z, z = wx + b
        we want to calculate dL/dw
        dL/da is known (grad_out)

        dL/dw = dL/dz * dz/dw = dL/dz * x
        dL/db = dL/dz * dz/db = dL/dz

        dL/dx = dL/dz * dz/dx = dL/dz * w
        """
        self.dw = np.outer(grad_out, self.x)
        self.db = grad_out

        self._optimize()

        dx = grad_out @ self.w
        return dx
    
    def _optimize(self):
        self.log(f"weights before: {self.w}")
        self.log(f"bias before: {self.b}")

        self.w = self.w_optimizer.optimize(self.w, self.dw)
        self.b = self.b_optimizer.optimize(self.b, self.db)

        self.log(f"weights after: {self.w}")
        self.log(f"bias after: {self.b}")


    def set_optimizer(self, optimizer):
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer

        elif isinstance(optimizer, str):
            self.optimizer = OptimizerBuilder.build(optimizer)



if __name__ == "__main__":
    model_debug = Linear(3, 5)

    x = np.random.randn(3)

    print(f"x: {x}")

    model_debug.forward(x)





