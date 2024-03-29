import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Compute the updated weight tensor by subtracting the product of learning rate and gradient tensor from the weight tensor
        # w1=w1-learing_rate*d(error)/dw1
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight


class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_gradient = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.prev_gradient = self.prev_gradient * self.momentum_rate - self.learning_rate * gradient_tensor
        updated_weight = self.prev_gradient + weight_tensor
        return updated_weight


class Adam:
    def __init__(self, learning_rate: float, mu: float, rho: float):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.vk = 0
        self.rk = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        self.vk = self.mu * self.vk + (1 - self.mu) * gk
        self.rk = self.rho * self.rk + (1 - self.rho) * (gk ** 2)
        rcap = self.rk / (1 - self.rho ** self.k)
        vcap = self.vk / (1 - self.mu ** self.k)
        self.k += 1
        gradient_tensor = vcap / (rcap ** 0.5 + np.finfo(float).eps)
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight