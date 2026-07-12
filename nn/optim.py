import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from nn import Log


class Optimizer(Log, ABC):
    def __init__(self, *params, lr: float = 0.01, verbose=False):
        """
        Initialize optimizer
        Args:
            *params: parameters to be optimized OR tuples of the parameters (np.ndarray) to be optimized
        """
        super().__init__(verbose)

        if len(params) == 1 and isinstance(params[0], dict):
            self.params = params[0]
        elif isinstance(params, tuple) and all(isinstance(param, np.ndarray) for param in params):
            self.params = {
                id(param): param for param in params
            }
        else:
            raise ValueError(f"unexpected params type! Expected: dict[int, np.ndarray] or tuple[np.ndarray], got: {type(params)}")
        self.lr: float = lr

    def step(self, gts):
        """
        Args:
            gts: dict of np.ndarray
                 gradients keyed by the same ids as self.params.
                 Each must be the same shape as the corresponding theta.
        """
        for id_, gt in gts.items():
            if id_ not in self.params.keys():
                raise KeyError(f"No id {id_} in params")

            self.update_params(id_, gt)

    @abstractmethod
    def update_params(self, id_, gt):
        raise NotImplementedError("Not implemented in base class")
    
    def see_param(self):
        for param in self.params:
            print(param)


class GradientDescent(Optimizer):
    def __init__(self, *params, lr=0.01, verbose=False):
        super().__init__(*params, lr=lr, verbose=verbose)

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        theta -= self.lr * gt


class Momentum(Optimizer):
    def __init__(self, *params, lr=0.01, gamma=0.01):
        super().__init__(*params, lr=lr)
        self.gamma = gamma
        self.vt = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.vt[id_] = (self.gamma * self.vt[id_]) + (self.lr * gt)
        theta -= self.vt[id_]


class AdaGrad(Optimizer):
    def __init__(self, *params, lr=0.5, epsilon=1e-4):
        super().__init__(*params, lr=lr)
        self.epsilon = epsilon
        self.vt = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.vt[id_] += gt ** 2
        theta -= (self.lr / (np.sqrt(self.vt[id_]) + self.epsilon)) * gt


class RMSProp(Optimizer):
    def __init__(self, *params, lr=0.01, gamma=0.99, epsilon=1e-8):
        super().__init__(*params, lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.vt = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.vt[id_] = (self.gamma * self.vt[id_]) + ((1 - self.gamma) * (gt ** 2))
        theta -= (self.lr / (np.sqrt(self.vt[id_]) + self.epsilon)) * gt


class Adam(Optimizer):
    def __init__(self, *params, lr=0.5, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
        super().__init__(*params, lr=lr)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.mt = {id_: 0 for id_ in self.params.keys()}
        self.vt = {id_: 0 for id_ in self.params.keys()}
        self.t = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.t[id_] += 1
        t = self.t[id_]

        self.mt[id_] = self.beta_1 * self.mt[id_] + ((1 - self.beta_1) * gt)
        self.vt[id_] = self.beta_2 * self.vt[id_] + ((1 - self.beta_2) * (gt ** 2))

        mt_hat = self.mt[id_] / (1 - (self.beta_1 ** t))
        vt_hat = self.vt[id_] / (1 - (self.beta_2 ** t))

        theta -= (self.lr / (np.sqrt(vt_hat) + self.epsilon)) * mt_hat


if __name__ == "__main__":
    target_w = 1_000
    target_b = 1_000

    x = np.random.randn(100)
    y = target_w * x + np.random.randn(100) * 0.1 + target_b

    # w is the learnable parameter
    w = np.random.randn(1)
    b = np.random.randn(1)

    optimizer = Adam({
        id(w): w,
        id(b): b
    })
    
    for step_i in range(10_000):
        y_pred = w * x + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        
        grad_w = np.mean(2 * error * x)
        grad_w = np.array([grad_w])    
        
        grad_b = np.mean(2 * error)
        grad_b = np.array([grad_b])
        
        optimizer.step(
            {
                id(w): grad_w,
                id(b): grad_b
             }
            )

        if step_i % 20 == 0:
            print(f"step {step_i:3d} | loss: {loss:.6f} | w: {w[0]:.4f} | b: {b[0]:.4f}")

