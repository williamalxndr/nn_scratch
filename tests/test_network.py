import pytest
from nn.network import Network
import numpy as np
from nn.linear import Linear

### Test cases for network

@pytest.mark.parametrize("input_size",  range(3, 7, 3))    
@pytest.mark.parametrize("output_size", range(1, 10, 5))   
@pytest.mark.parametrize("batch_size",  range(1, 4))   
@pytest.mark.parametrize("epochs",      range(100, 2000, 1000)) 
@pytest.mark.parametrize("training_batch_size", range(1, 4))
def test_network_learning(input_size, output_size, batch_size, epochs, training_batch_size):
    if training_batch_size > batch_size:
        pytest.skip("training_batch_size > batch_size")
    
    net = Network(loss_type="mse", epochs=epochs, training_batch_size=training_batch_size)
    net.add_layer(Linear(input_size, 10))
    net.add_layer(Linear(10, output_size))

    x = np.random.randn(batch_size, input_size)
    y = np.random.randn(batch_size, output_size)
    losses = net.train(x, y)

    assert(losses[-1] < losses[0])


    

