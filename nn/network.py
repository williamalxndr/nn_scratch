import numpy as np
from nn.base_class import Layer
from loss import *


class Network:
    def __init__(self, *layers, loss_type, epoch):
        self.layers = []

        # Add each layer to network
        for layer in layers:
            self.add_layer(layer)

        LOSS_METHOD_DICT = {
            "mse": mse,
            "mean_squared_error": mse,
            "bce": bce,
            "binary_cross_entropy": bce
        }

        self.loss_method = LOSS_METHOD_DICT[self.loss_type]
        self.epoch = epoch


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

            self.log(f"Layer: {layer}, output: {out}")

        self.out = out


    def backward(self, y_pred) -> Loss:
        loss = self.loss_method(self.out, y_pred)
        grad = loss.backward()

        for layer in self.layers:
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

            

if __name__ == "__main__":
    print("Hello world")
