import numpy as np

from base_class import Optimizer



class GradientDescent(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def optimize(self, dtheta):
        dtheta -= self.lr * dtheta
        return dtheta
    

class Momentum(Optimizer):
    def __init__(self, lr=0.01, gamma=0.01):
        super().__init__(lr)
        self.gamma = gamma
        self.vt = 0

    def optimize(self, dtheta):
        gt = dtheta 

        self.vt += self.gamma * self.vt + self.lr * gt
        dtheta -= self.vt

        return dtheta


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-4):
        super().__init__(lr)
        self.epsilon = epsilon
        self.vt = 0
        

    def optimize(self, dtheta):
        gt = dtheta

        self.vt += gt ** 2
        dtheta -= (self.lr / (np.sqrt(self.vt) + self.epsilon)) * gt

        return dtheta
        

class RMSProp(Optimizer):
    def __init__(self, lr=1e-2, gamma=1e-3, epsilon=1e-4):
        super().__init__(lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.vt = 0

    def optimize(self, dtheta):
        gt = dtheta

        self.vt = self.gamma * self.vt + (1-self.gamma) * (gt**2)
        dtheta -= (self.lr / (np.sqrt(self.vt) + self.epsilon)) * gt

        return dtheta

    
class Adam(Optimizer):
    def __init__(self, lr=1e-2, epsilon=1e-4, beta_1=1e-3, beta_2=1e-3):
        super().__init__(lr)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.mt = 0
        self.vt = 0
        self.t = 0

    def optimize(self, dtheta):
        self.t += 1

        gt = dtheta

        self.mt = self.beta_1 * self.mt + ((1 - self.beta_1) * gt)
        self.vt = self.beta_2 * self.vt + ((1 - self.beta_2) * (gt**2))
        
        mt_head = self.mt / (1 - (self.beta_1**self.t))
        vt_head = self.vt / (1 - (self.beta_2**self.t))

        dtheta -= (self.lr / (np.sqrt(vt_head) + self.epsilon)) * mt_head

        return dtheta


class OptimizerBuilder:
    def __init__(self):
        """
        Builder for optimizer
        """
        pass

    def build(self, type):
        if type.lower() == "gradient descent":
            return GradientDescent()
        elif type.lower() == "momentum": 
            return Momentum()
        elif type.lower() == "adagrad":
            return AdaGrad()
        elif type.lower() == "rmsprop":
            return RMSProp()
        elif type.lower() == "adam":
            return Adam()
        
        