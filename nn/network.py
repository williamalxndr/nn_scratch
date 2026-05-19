import numpy as np
from base_layer import Layer
from loss import *


class Network:
    def __init__(self, *layers, loss, epoch):
        self.layers = []

        # Add each layer to network
        for layer in layers:
            self.add_layer(layer)

        if loss.lower() == "mse" or loss.lower() == "mean_squared_error":
            self.loss_with_grad = mse_with_grad
            self.loss = mse

        if loss.lower() == "bce" or loss.lower() == "binary_cross_entropy":
            self.loss_with_grad = binary_cross_entropy_with_grad
            self.loss = binary_cross_entropy

        self.epoch = epoch


    def add_layer(self, layer: Layer):
        """
        Add layer to the last position in network
        """
        self.layers.append(layer)


    def forward(self, x):
        out = x

        for layer in self.layers:
            out = layer.forward(out)

        self.out = out


    def backward(self, y_pred):
        _, grad = self.loss_with_grad(self.out, y_pred)

        for layer in self.layers:
            grad = layer.backward(grad)

    
    def train(self, x, y):
        for _ in range(self.epoch):
            self.forward(x)
            self.backward(y)




            
if __name__ == "__main__":
    print("Hello world")
