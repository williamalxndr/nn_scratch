import numpy as np


# =============== MEAN SQUARED ERROR ===============
def mse(y_pred, y_true):
    """
    Calculate the Mean Squared Error
    MSE = ||y_true - y_pred||^2


    --- Input ---
    y_pred: predicted value
    y_true: target value

    --- Output --- 
    Returns: loss
    loss: MSE of y_pred
    """
    if y_pred.shape != y_true.shape:
        msg = f"\Shape mismatched! \
            y_pred 's shape: {y_pred.shape}, y_true's shape: {y_true.shape}"
        raise ValueError(msg)

    loss = 0.5 * ((y_true - y_pred) ** 2)
    return loss


def mse_with_grad(y_pred, y_true):
    """
    Calculate the Mean Squared Error and gradient of the MSE loss
    MSE = ||y_true - y_pred||^2

    Used for backpropagation

    --- Input ---
    y_pred: predicted value
    y_true: target value

    --- Output --- 
    Returns: loss, grad_out
    loss: MSE of y_pred
    grad_out: gradient of the loss with respect to y_pred
    """
    loss = mse(y_pred, y_true)
    grad_out = y_true - y_pred
    return loss, grad_out


# =============== BINARY CROSS ENTROPY ===============

def binary_cross_entropy(y_pred, y_true):
    """
    Calculate the Binary Cross Entropy
    BCE = -1/N * sum{y_i * log(p_i) + ((1-y_i) * log (1-p_i))}

    --- Input ---
    y_pred: target value
    y_true: target value

    --- Output ---
    Returns: loss
    loss: BCE of y_pred
    """
    if y_pred.shape != y_true.shape:
        msg = f"\Shape mismatched! \
            y_pred 's shape: {y_pred.shape}, y_true's shape: {y_true.shape}"
        raise ValueError(msg)

    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def binary_cross_entropy_with_grad(y_pred, y_true):
    """
    Calculate the Binary Cross Entropy and gradient
    BCE = -1/N * sum{y_i * log(p_i) + ((1-y_i) * log (1-p_i))}

    --- Input ---
    y_pred: target value
    y_true: target value

    --- Output ---
    Returns: loss, grad_out
    loss: BCE of y_pred
    grad_out: Gradient of BCE with respect to y_pred
    """
    loss = binary_cross_entropy(y_pred, y_true)
    grad_out = np.mean((-y_true/y_pred) + ((1-y_true)/(1-y_pred)))
    return loss, grad_out

