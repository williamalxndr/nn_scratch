import numpy as np
from nn import Layer
from nn.loss import *
from nn.linear import Linear
from nn.optim import *
from argparse import ArgumentParser


class Network(Layer):
    def __init__(self, *layers: Layer):
        super().__init__(input_size=None, output_size=None, verbose=False)
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

            # Check vanishing/exploding gradient
            if np.any(grad_out> 1e10):
                self.log("Gradient explodes", True)
            elif np.any(grad_out < 1e-10):
                self.log("Gradient vanishes", True)

    def _add_layer(self, layer: Layer):
        """
        Add layer to the last position in network
        """
        if not isinstance(layer, Layer):
            raise ValueError("layer must be Layer object")
        
        self.layers.append(layer)


            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=3)
    parser.add_argument("--training_batch_size", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="Adam")
    args = parser.parse_args()
    
    net = Network(Linear(args.input_size,5), Linear(5,2), Linear(2,output_size=args.output_size))
    
    optimizer = Adam(net)
    
    print(net.parameters())

