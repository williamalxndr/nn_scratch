import torch
import numpy as np
import pytest

from nn.activation_function import *
from nn.linear import Linear

builder = ActivationFunctionBuilder()

@pytest.mark.parametrize("batch_size",  range(1, 100, 20)) 
@pytest.mark.parametrize("size",  range(5, 21, 5)) 
@pytest.mark.parametrize("type", ["Sigmoid", "ReLU"])
def test_activation_function(batch_size, size, type):
    act = builder.build(type, size)
    x = np.random.randn(batch_size, size)
    x_torch = torch.from_numpy(x)

    if type == "Sigmoid":
        act_torch = torch.nn.functional.sigmoid
    elif type == "ReLU":
        act_torch = torch.nn.functional.relu

    y = act(x)
    y_torch = act_torch(x_torch)

    assert np.allclose(y, y_torch)




