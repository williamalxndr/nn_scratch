import pytest
from nn.network import Network
import numpy as np
from nn.linear import Linear

### Test cases for network

@pytest.mark.parametrize("input_size",  range(3, 9, 3))      # [3, 6] → 2
@pytest.mark.parametrize("output_size", range(1, 51, 10))    # [1, 11, 21, 31, 41] → 5
@pytest.mark.parametrize("batch_size",  range(1, 26, 5))     # [1, 6, 11, 16, 21] → 5
@pytest.mark.parametrize("epochs",      range(100, 600, 400)) # [100, 500] → 2
def test_network_learning(input_size, output_size, batch_size, epochs):
    net = Network(loss_type="mse", epoch=epochs)
    net.add_layer(Linear(input_size, 10))
    net.add_layer(Linear(10, output_size))

    x = np.random.randn(input_size, batch_size)
    y = np.random.randn(output_size, batch_size)
    losses = net.train(x, y)

    print("Loss 0 ")
    print(losses[0])
    print("Loss -1")
    print(losses[-1])

    assert(losses[-1] > losses[0])
    

