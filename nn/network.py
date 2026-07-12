import numpy as np
from nn import Layer
from nn.loss import *
from nn.linear import Linear
from nn.optim import *
from argparse import ArgumentParser


class Network(Layer):
    def __init__(self, *layers: Layer, loss_type: str, epochs: int, optimizer=Adam, training_batch_size=1):
        super().__init__(input_size=None, output_size=None, verbose=False)
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
        self.epochs = epochs
        self.training_batch_size = training_batch_size


    def set_optimizer(self, optimizer):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.set_optimizer(optimizer)

    def add_layer(self, layer: Layer):
        """
        Add layer to the last position in network
        """
        if not isinstance(layer, Layer):
            raise ValueError("layer must be Layer object")
        
        self.layers.append(layer)


    def forward(self, x: np.ndarray):
        """
        Performing forward pass through the layers in the network

        Args:
            x: input array, np.ndarray with shape (batch_size, input_size)

        Returns:
            y: output array, np.ndarray with shape (batch_size, output_size) 
        """
        out = x

        for layer in self.layers:
            out = layer(out)

        self.out = out

        return self.out


    def backward(self, y_true: np.ndarray) -> Loss:
        loss = self.loss_method(self.out, y_true)
        grad = loss.backward()

        for layer in reversed(self.layers):            
            grad = layer.backward(grad)

            # Check vanishing/exploding gradient
            if np.any(grad> 1e10):
                self.log("Gradient explodes", True)
            elif np.any(grad < 1e-10):
                self.log("Gradient vanishes", True)


        return loss

    
    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the neural network for n epochs
        Args:
            x: training data, np.ndarray with shape (batch_size, input_size)
            y: target data, np.ndarray with shape (batch_size, output_size)

        Returns:
            losses: Losses of the training process, np.ndarray with shape (batch_size, )

        """
        losses = []
        
        if x.shape[0] != y.shape[0]:
            raise ValueError("Batch size mismatched!")
        
        row = x.shape[0]

        for i in range(self.epochs): 
            # Set batch training data
            batch_idx = np.random.choice(np.arange(row, dtype=np.int32), size=self.training_batch_size, replace=False)
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            print(x.shape[0])

            self.log(f"Batch_idx: {batch_idx}", force=True)

            self.log(f"x: {x}")
            self.log(f"y: {y}")

            self.log(f"x_batch: {x_batch}", force=True)  
            self.log(f"y_batch: {y_batch}", force=True)

            self.forward(x_batch)
            loss = self.backward(y_batch)

            losses.append(loss.get_mean())

            if i % 100:
                self.log(f"Loss at epoch {i}: {loss.get_mean()}", force=True)

        return losses



            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=3)
    parser.add_argument("--training_batch_size", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="Adam")
    args = parser.parse_args()
    
    net = Network(Linear(args.input_size,5), Linear(5,2), Linear(2,output_size=args.output_size), loss_type="mse", epochs=args.epochs, training_batch_size=args.training_batch_size)

    net.set_optimizer(args.optimizer)

    x = np.random.randn(args.batch_size, args.input_size)
    y_true = np.random.randn(args.batch_size, args.output_size)

    net.train(x, y_true)
