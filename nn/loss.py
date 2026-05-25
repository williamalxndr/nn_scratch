import numpy as np


class Loss:
    def __init__(self, value: np.ndarray, grad: np.ndarray):
        """
        Initiate loss object.

        Args:
            value: loss values, np.ndarray with shape (batch_size, output_size)
            grad: dL/dy, np.ndarray with shape (batch_size, output_size)
        """
        self.value = value
        self.grad = grad

    def __str__(self):
        return str(self.value)
        
    def backward(self):
        return self.grad
    
    def get_mean(self):
        return np.mean(self.value)
    
    def get_value(self):
        return self.value



# =============== MEAN SQUARED ERROR ===============
def mse(y_pred: np.ndarray, y_true: np.ndarray):
    """
    Calculate the Mean Squared Error

    Args:
        y_pred: np.ndarray with shape (batch_size, output_size)
        y_true: np.ndarray with shape (batch_size, output_size)

    Returns:
        Loss(loss_value, grad_out)

    Methods of Loss:
        def backward(self):  --> Returns the grad of loss w.r.t to y_pred
            return self.grad  
        
        def get_mean(self):  --> Returns the mean of the MSE for each batch
            return np.mean(self.value)  

        def get_value(self):  --> Returns the loss of each batch in array with shape (batch_size, 1)
            return self.value
    """
    
    if y_pred.shape != y_true.shape:
        msg = f"\Shape mismatched! \
            y_pred 's shape: {y_pred.shape}, y_true's shape: {y_true.shape}"
        raise ValueError(msg)

    loss_value = np.mean(((y_pred - y_true) ** 2), axis=1).reshape(-1,1)
    grad_out = 2/y_pred.shape[1] * (y_pred - y_true)

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

