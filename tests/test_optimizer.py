import pytest
import numpy as np

from nn.linear import Linear
from nn.loss import *
from nn.optimizer import *

builder = OptimizerBuilder()
TOL = 1e-1

### Helper function
def quadratic_function(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:    
    """
    Quadratic function of theta
    f(x) = x^2
    f'(x) or df/x = 2x
    Returns the squared and the gradient of the array
    """
    y = np.square(x)
    grad = 2 * x
    return y, grad

def shifted_quadratic_function(x:np.ndarray, k:float) -> tuple[np.ndarray, np.ndarray]:
    """
    natural logarithm function
    f(x) = (x - k)^2
    f'(x) = 2 (x - k)
    Returns tuple (f(x), f'(x))
    """ 
    k = k * np.ones_like(x)
    y = np.square(x - k)
    grad = 2 * (x - k)
    return y, grad


@pytest.mark.parametrize("batch_size", range(1, 102, 20))
@pytest.mark.parametrize("output_size", range(1, 22, 5))
@pytest.mark.parametrize("iteration", range(1000, 2000, 500))
@pytest.mark.parametrize("optimizer_type", ["Gradient Descent", "Momentum", "AdaGrad", "RMSProp", "Adam"])
def test_quadratic(optimizer_type, batch_size, output_size, iteration):
    """
    f(x) = x^2
    the minimum should be x = 0
    """
    optimizer = builder.build(optimizer_type)
    theta = np.random.randn(batch_size, output_size)
    _, dtheta = quadratic_function(theta)
    for _ in range(iteration):
        theta = optimizer.optimize(theta, dtheta)
        _, dtheta = quadratic_function(theta)

    print(f"theta: {theta}") 
    assert np.allclose(theta, np.zeros_like(theta), atol=TOL)
        
        


@pytest.mark.parametrize("batch_size", range(1, 102, 50))
@pytest.mark.parametrize("output_size", range(1, 22, 10))
@pytest.mark.parametrize("iteration", range(10000, 20000, 5000))
@pytest.mark.parametrize("k", range(1, 21, 10))
@pytest.mark.parametrize("optimizer_type", ["Gradient Descent", "Momentum", "AdaGrad", "RMSProp", "Adam"])
def test_shifted_quadratic_function(optimizer_type, batch_size, output_size, k, iteration):
    """
    f(x) = (x-k)^2
    the minimum should be x = k, f(x) = 0
    """
    optimizer = builder.build(optimizer_type)
    theta = np.random.randn(batch_size, output_size)
    _, dtheta = shifted_quadratic_function(theta, k)
    for _ in range(iteration):
        theta = optimizer.optimize(theta, dtheta)
        _, dtheta = shifted_quadratic_function(theta, k)

    print(f"theta: {theta}")

    assert np.allclose(theta, k * np.ones_like(theta), atol=TOL)
        
        


    
    