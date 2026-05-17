from nn.linear import Linear
import numpy as np

if __name__ == "__main__":
    layer_1 = Linear(3, 5, verbose=True)
    layer_2 = Linear(5, 2, verbose=True)

    x = np.random.randn(3)

    y_true = np.random.randn(2)

    y_pred = layer_2.forward(layer_1.forward(x))

    loss = 0.5 * ((y_true - y_pred) ** 2)
    grad_loss = y_true - y_pred

    print(f"Grad loss: {grad_loss}")


    layer_1.backward(layer_2.backward(grad_loss))

    