import torch
import pytest
from torch.nn.functional import mse_loss
import numpy as np

from nn.loss import *
from nn.linear import Linear



@pytest.mark.parametrize("output_size", range(1, 100, 10))
@pytest.mark.parametrize("batch_size", range(1, 2000, 200))
def test_mse_loss(batch_size, output_size):
    y_pred = np.random.randn(batch_size, output_size)
    y_true = np.random.randn(batch_size, output_size)

    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)

    loss = mse(y_pred, y_true).get_mean()
    loss_torch = mse_loss(y_pred_torch, y_true_torch).item()

    assert np.isclose(loss, loss_torch)





