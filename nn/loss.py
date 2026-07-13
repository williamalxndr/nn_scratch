import numpy as np
from abc import ABC, abstractmethod
from nn import Layer

_EPS = 1e-12


class LossLayer(Layer):
    """Base class for losses as terminal layers in the network."""

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self._grad = None

    @abstractmethod
    def _compute(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Return (loss_value, grad_out)."""
        raise NotImplementedError()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Args:
            y_pred: np.ndarray with shape (batch_size, output_size)
            y_true: np.ndarray with shape (batch_size, output_size)

        Returns:
            loss_value: np.ndarray with shape (batch_size, 1)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatched! y_pred's shape: {y_pred.shape}, "
                f"y_true's shape: {y_true.shape}"
            )
        loss_value, grad_out = self._compute(y_pred, y_true)
        self._grad = grad_out
        return loss_value

    def backward(self, grad_out=1.0) -> np.ndarray:
        """Returns dL/dy_pred, shape (batch_size, output_size)."""
        return self._grad

    def parameters(self):
        return {}

    def grads(self):
        return {}
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def get_mean(self) -> float:
        """Returns the mean of the loss over the batch."""
        return np.mean(self._grad is not None and self.value).item()


class MSE(LossLayer):
    """Mean Squared Error loss layer."""

    def _compute(self, y_pred, y_true):
        loss_value = np.mean((y_pred - y_true) ** 2, axis=1).reshape(-1, 1)
        grad_out = 2 / y_pred.shape[1] * (y_pred - y_true)
        return loss_value, grad_out


class BCE(LossLayer):
    """Binary Cross Entropy loss layer."""

    def _compute(self, y_pred, y_true):
        p = np.clip(y_pred, _EPS, 1 - _EPS)
        loss_value = -np.mean(
            y_true * np.log(p) + (1 - y_true) * np.log(1 - p), axis=1
        ).reshape(-1, 1)
        grad_out = (1 / y_pred.shape[1]) * (p - y_true) / (p * (1 - p))
        return loss_value, grad_out