import pytest
import numpy as np
from nn.linear import Linear
from nn.loss import *

### Test cases for network

@pytest.mark.parametrize("input_size",  range(4, 10, 3))   
@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_forward_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)

    x = np.random.randn(batch_size, input_size)
    y = linear.forward(x)
    
    assert y.shape == (batch_size, output_size)


@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_pred = linear.forward(x)

    y_true = np.random.randn(output_size)

    loss = mse(y_pred, y_true)
    assert loss.backward().shape == (batch_size, output_size)

    grad = linear.backward(loss.backward())

    assert linear.dw.shape == (output_size, input_size)
    assert linear.db.shape == (output_size, 1)


    assert grad.shape == (batch_size, input_size)
    




