import pytest
import numpy as np
from nn.linear import Linear
from nn.loss import *

### Test cases for network

@pytest.mark.parametrize("input_size",  range(4, 10, 3))   
@pytest.mark.parametrize("output_size", range(1, 51, 10)) 
@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
def test_forward_type(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)

    x = np.random.randn(batch_size, input_size)
    y = linear.forward(x)
    
    assert isinstance(y, np.ndarray)


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
def test_backward_dl_dw_type(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    y_pred = linear.forward(x)

    loss = mse(y_pred, y_true)
    linear.backward(loss.backward())

    assert isinstance(linear.dw, np.ndarray)

@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_dl_db_type(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    y_pred = linear.forward(x)

    loss = mse(y_pred, y_true)
    linear.backward(loss.backward())

    assert isinstance(linear.db, np.ndarray)

@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_dl_db_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    y_pred = linear.forward(x)

    loss = mse(y_pred, y_true)
    linear.backward(loss.backward())

    assert linear.dw.shape == (output_size, input_size)

@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_dl_db_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    y_pred = linear.forward(x)

    loss = mse(y_pred, y_true)
    linear.backward(loss.backward())

    assert linear.db.shape == (1, output_size)


@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_type(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    y_pred = linear.forward(x)

    loss = mse(y_pred, y_true)
    dl_dx = linear.backward(loss.backward())

    assert isinstance(dl_dx, np.ndarray)


@pytest.mark.parametrize("input_size", range(3, 12, 3))
@pytest.mark.parametrize("output_size", range(1, 40, 5))
@pytest.mark.parametrize("batch_size", range(1, 100, 10))
def test_backward_shape(input_size, output_size, batch_size):
    linear = Linear(input_size, output_size)
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    y_pred = linear.forward(x)

    loss = mse(y_pred, y_true)
    dl_dx = linear.backward(loss.backward())

    assert dl_dx.shape == (batch_size, input_size)


def test_optimizer():
    pass