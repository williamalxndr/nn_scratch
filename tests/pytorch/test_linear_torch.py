import torch
import pytest
from torch.nn.functional import mse_loss
import numpy as np

from nn.loss import *
from nn.linear import Linear

@pytest.mark.parametrize("input_size", range(1, 40, 5))
@pytest.mark.parametrize("output_size", range(1, 100, 20))
@pytest.mark.parametrize("batch_size", range(1, 10000, 2000))
def test_linear_output(input_size, output_size, batch_size):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    x_torch = torch.from_numpy(x)

    linear = Linear(input_size, output_size)
    linear_torch = torch.nn.Linear(input_size, output_size)

    linear.w = linear_torch.weight.detach().numpy()
    linear.b = linear_torch.bias.detach().numpy()


    y = linear.forward(x)
    y_torch = linear_torch.forward(x_torch).detach().numpy() 

    assert np.allclose(y, y_torch, atol=1e-5, rtol=1e-4)




    
