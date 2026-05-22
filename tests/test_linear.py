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

    x = np.random.randn(input_size, batch_size)
    y = linear.forward(x)
    
    assert y.shape == (output_size, batch_size)

@pytest.mark.parametrize("input_size",  range(3, 30, 6))   
@pytest.mark.parametrize("output_size", range(1, 100, 10)) 
def test_forward_shape_with_x_ndim_1(input_size, output_size):
    linear = Linear(input_size, output_size)

    x = np.random.randn(input_size)
    y = linear.forward(x)

    assert y.shape == (output_size, 1)


@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(input_size, batch_size)
    y_pred = linear.forward(x)

    y_true = np.random.randn(output_size)

    loss = mse(y_pred, y_true)
    assert loss.backward().shape == (output_size, batch_size)

    grad = linear.backward(loss.backward())
    assert grad.shape == (input_size, batch_size)
    




