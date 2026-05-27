import numpy as np
import pytest

from nn.activation_function import *
from nn.linear import Linear

act_builder = ActivationFunctionBuilder()

@pytest.mark.parametrize("batch_size",  range(1, 26, 5)) 
@pytest.mark.parametrize("size",  range(5, 21, 5)) 
@pytest.mark.parametrize("type", ["Sigmoid", "ReLU"])
def test_shape(batch_size, size, type):
    act = act_builder.build(type, size)
    x = np.random.randn(batch_size, size)
    y = act.forward(x)

    assert y.shape == (batch_size, size)

