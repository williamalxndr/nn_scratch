from nn import Network, Linear
from nn.loss import MSE
from nn.optim import Adam
from nn.activation_function import *
import numpy as np

if __name__ == "__main__":
    net = Network(
        Linear(input_size=4, output_size=32, init="xavier"),
        Sigmoid(),
        Linear(input_size=32, output_size=256, init="xavier"),
        Sigmoid(),
        Linear(input_size=256, output_size=4, init="xavier"),
    )
    loss_fn = MSE()
    optimizer = Adam(net, lr=1e-2)

    x = np.random.randn(100, 4)
    y = 41.23 * x ** 3 + 312.459 * x ** 2 + 59.145 * x 

    for epoch in range(99999):
        y_pred = net(x)
        loss_value = loss_fn(y_pred, y)

        grad_out = loss_fn.backward()
        net.backward(grad_out)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss {loss_value.mean():.6f}")