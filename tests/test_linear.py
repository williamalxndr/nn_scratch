import pytest
from nn.network import Network
import numpy as np
from nn.linear import Linear

### Test cases for network

@pytest.mark.parametrize("input_size",  range(3, 10, 3))   
@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_forward_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)

    x = np.random.randn(input_size, batch_size)
    y = linear.forward(x)
    
    assert y.shape == (output_size, batch_size)

