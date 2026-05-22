import numpy as np


class Loss:
    def __init__(self, value: np.array, grad: np.array):
        self.value = value
        self.grad = grad

    def __str__(self):
        return str(self.value)
        
    def backward(self):
        return self.grad
    
    def get_mean(self):
        return np.mean(self.value)



# =============== MEAN SQUARED ERROR ===============
def mse(y_pred, y_true):
    """
    Calculate the Mean Squared Error and gradient of the MSE loss
    MSE = ||y_true - y_pred||^2

    Used for backpropagation

    --- Input ---
    y_pred: predicted value
    y_true: target value

    --- Output --- 
    Returns Loss(loss_value, grad_out)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if y_pred.shape != y_true.shape:
        msg = f"\Shape mismatched! \
            y_pred 's shape: {y_pred.shape}, y_true's shape: {y_true.shape}"
        raise ValueError(msg)

    loss_value = np.mean(((y_pred - y_true) ** 2), axis=0)
    grad_out = 2/y_pred.shape[0] * (y_pred - y_true)

    return Loss(loss_value, grad_out)


# =============== BINARY CROSS ENTROPY ===============
def bce(y_pred, y_true):
    """
    Calculate the Binary Cross Entropy and gradient
    BCE = -1/N * sum{y_i * log(p_i) + ((1-y_i) * log (1-p_i))}

    --- Input ---
    y_pred: target value
    y_true: target value

    --- Output ---
    Returns Loss(loss_value, grad_out)
    """
    if y_pred.shape != y_true.shape:
        msg = f"\Shape mismatched! \
            y_pred 's shape: {y_pred.shape}, y_true's shape: {y_true.shape}"
        raise ValueError(msg)

    loss_value = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    grad_out = np.mean((-y_true/y_pred) + ((1-y_true)/(1-y_pred)))
    return Loss(loss_value, grad_out)

