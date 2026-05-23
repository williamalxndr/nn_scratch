from nn.linear import Linear
from nn.loss import *

import numpy as np
import pytest

@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_mse_type(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    loss = mse(y_pred, y_true)

    assert isinstance(loss, Loss)


@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_mse_value_type(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    loss = mse(y_pred, y_true)

    loss_value = loss.get_value()
    assert isinstance(loss_value, np.ndarray)


@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_mse_value_shape(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    loss = mse(y_pred, y_true)

    loss_value = loss.get_value()
    assert loss_value.shape == (batch_size, 1)


@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_mse_mean_type(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    loss = mse(y_pred, y_true)

    loss_mean = loss.get_mean()
    assert isinstance(loss_mean, float)


@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_mse_grad_type(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    loss = mse(y_pred, y_true)

    grad = loss.backward()
    
    assert isinstance(grad, np.ndarray)



@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_mse_grad_shape(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    loss = mse(y_pred, y_true)

    grad = loss.backward()
    
    assert grad.shape == (batch_size, output_size)



