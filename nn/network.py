import numpy as np
from base_class import Layer
from loss import *
from linear import Linear
from optimizer import *


class Network:
    def __init__(self, *layers: Layer, loss_type: str, epoch: int, optimizer=Adam, verbose=True):
        self.layers = []

        # Add each layer to network
        for layer in layers:
            if isinstance(layer, Linear):
                layer.set_optimizer(optimizer)
            self.add_layer(layer)

        LOSS_METHOD_DICT = {
            "mse": mse,
            "mean_squared_error": mse,
            "bce": bce,
            "binary_cross_entropy": bce
        }

        self.loss_method = LOSS_METHOD_DICT[loss_type]
        self.epoch = epoch
        self.verbose=verbose


    def set_optimizer(self, optimizer):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.set_optimizer(optimizer)

    def add_layer(self, layer: Layer):
        """
        Add layer to the last position in network
        """
        self.layers.append(layer)


    def forward(self, x: np.ndarray):
        """
        Forward pass through the network

        """
        out = x

        for layer in self.layers:
            out = layer(out)

            # self.log(f"Layer: {layer}, output: {out}")

        self.out = out


    def backward(self, y_true) -> Loss:
        loss = self.loss_method(self.out, y_true)
        grad = loss.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    
    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the neural network for n epochs

        """
        for _ in range(self.epoch):
            self.forward(x)
            loss = self.backward(y)

            self.log(f"Loss at epoch {_}: {loss}")


    def log(self, msg):
        if self.verbose:
            print(msg)

    

            

if __name__ == "__main__":
    OUTPUT_SIZE = 4
    ROW = 3
    EPOCH = 1000
    net = Network(Linear(3,5), Linear(5,2), Linear(2,OUTPUT_SIZE), loss_type="mse", epoch=1000)

    net.set_optimizer(Adam)

    x = np.random.randn(3, ROW)
    y_true = np.random.randn(OUTPUT_SIZE, ROW)

    net.train(x, y_true)

