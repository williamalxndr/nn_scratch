import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from nn import Log, Layer


class Optimizer(Log, ABC):
    def __init__(self, *layers, lr=0.01, verbose=False):
        super().__init__(verbose)
        if not all(isinstance(layer, Layer) for layer in layers):
            raise ValueError("layers should be Layer")
        self.layers = layers
        self.lr = lr
        
        self.params = {}
        for layer in self.layers:
            self.params.update(layer.parameters())
        
        self.lr = lr

    def step(self):
        for layer in self.layers:
            for id_, gt in layer.grads().items():
                if id_ not in self.params:
                    raise KeyError(f"No id {id_} in params")
                self.update_params(id_, gt)
            
    @abstractmethod
    def update_params(self, id_, gt):
        raise NotImplementedError("Not implemented in base class")
    
    def see_param(self):
        for param in self.params:
            print(param)


class GradientDescent(Optimizer):
    def __init__(self, *layers, lr=0.01, verbose=False):
        super().__init__(*layers, lr=lr, verbose=verbose)

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        theta -= self.lr * gt


class Momentum(Optimizer):
    def __init__(self, *layers, lr=0.01, gamma=0.01):
        super().__init__(*layers, lr=lr)
        self.gamma = gamma
        self.vt = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.vt[id_] = (self.gamma * self.vt[id_]) + (self.lr * gt)
        theta -= self.vt[id_]


class AdaGrad(Optimizer):
    def __init__(self, *layers, lr=0.5, epsilon=1e-4):
        super().__init__(*layers, lr=lr)
        self.epsilon = epsilon
        self.vt = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.vt[id_] += gt ** 2
        theta -= (self.lr / (np.sqrt(self.vt[id_]) + self.epsilon)) * gt


class RMSProp(Optimizer):
    def __init__(self, *layers, lr=0.01, gamma=0.99, epsilon=1e-8):
        super().__init__(*layers, lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.vt = {id_: 0 for id_ in self.params.keys()}

    def update_params(self, id_, gt):
        theta = self.params.get(id_)
        self.vt[id_] = (self.gamma * self.vt[id_]) + ((1 - self.gamma) * (gt ** 2))
        theta -= (self.lr / (np.sqrt(self.vt[id_]) + self.epsilon)) * gt


class Adam(Optimizer):
    def __init__(self, *layers, lr=0.5, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
        super().__init__(*layers, lr=lr)
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
    target_w1 = 634
    target_w2 = 1259
    target_b = 194

    x = np.random.randn(100)
    y = target_w1 * (x**2) + target_w2 * x + np.random.randn(100) * 0.1 + target_b

    # learnable parameter
    w1 = np.random.randn(1)
    w2 = np.random.randn(1)
    b = np.random.randn(1)

    optimizer = Adam({
        id(w1): w1,
        id(w2): w2,
        id(b): b
    })
    
    for step_i in range(10_000):
        y_pred = w1 * x**2 + w2 * x + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        
        grad_w1 = np.array([np.mean(2 * error * (x**2))])
        grad_w2 = np.array([np.mean(2 * error * x )])
        grad_b = np.array([np.mean(2 * error)])
        
        
        optimizer.step(
            {
                id(w1): grad_w1,
                id(w2): grad_w2,
                id(b): grad_b
             }
            )

        if step_i % 20 == 0:
            print(f"step {step_i:3d} | loss: {loss:.6f} | w1: {w1[0]:.4f} | w2: {w2[0]:.4f} | b: {b[0]:.04f}")

