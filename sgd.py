"""
Gradient based optimization methods

by Bowen, March 2018
"""


import numpy as np 
from autograd import elementwise_grad

class GradientBasedOptimization():

    def __init__(self, optimizer=None):
        self.Lambda = 0.01

        self.gamma = 0.9
        self.epsilon = 1e-8

        # Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1_exp = 1.0
        self.beta2_exp = 1.0
        
        # Adadelta & RMSprop params
        self.grad_expect = np.matrix(np.zeros((2,1)))
        self.delta_expect = np.matrix(np.zeros((2,1)))

        if optimizer == 'Adam':
            self.eta = 0.05
        elif optimizer == "Nesterov" or optimizer=="SGLD-Nesterov":
            self.eta = 0.0001
        else:
            self.eta = 0.001

        # SGLD params
        if optimizer == "Nesterov" or optimizer=="SGLD-Nesterov":
            self.sigma = 0.002
        elif optimizer == "SGLD":
            self.sigma = 0.008
        else:
            self.sigma = 0.005


        self.optimizer = optimizer


    def train(self, func, x0, y0, xmin, xmax, ymin, ymax, minima):
        x0s = [x0]
        y0s = [y0]

        minimum = False
        init = True
        steps = 0

        w = np.matrix(np.array([x0, y0]).reshape((2,1)))
        v = np.matrix(np.zeros(w.shape))
        m = np.matrix(np.zeros(w.shape))
        grad_sum_square = np.matrix(np.zeros(w.shape))    # for adagrad

        while x0 >= xmin and x0 <= xmax and y0 >= ymin and y0 <= ymax and not minimum and steps<20000:

            if self.optimizer == 'SGD':
                # Stochastic gradient descent
                grad = self.gradient(func, w)
                w = w - self.eta * grad
            elif self.optimizer == 'SGLD':
                # Stochastic gradient langevin dynamics
                grad = self.gradient(func, w)
                w = w - self.eta * grad + np.random.randn() * self.sigma
            elif self.optimizer == 'Momentum':
                # Momentum
                grad = self.gradient(func, w)
                v = self.gamma * v + self.eta * grad
                w = w - v
            elif self.optimizer == 'SGLD-Momentum':
                grad = self.gradient(func, w)
                v = self.gamma * v + self.eta * grad + np.random.randn() * self.sigma
                w = w - v
            elif self.optimizer == 'Nesterov':
                # Nesterov accelerated gradient
                w = w - self.gamma * v
                grad = self.gradient(func, w)
                v = self.gamma * v + self.eta * grad
                w = w - v
            elif self.optimizer == 'SGLD-Nesterov':
                w = w - self.gamma * v
                grad = self.gradient(func, w)
                v = self.gamma * v + self.eta * grad + np.random.randn() * self.sigma
                w = w - v
            elif self.optimizer == 'Adam':
                # Adam
                grad = self.gradient(func, w)
                dz_dx = elementwise_grad(func, argnum=0)(x0, y0)
                dz_dy = elementwise_grad(func, argnum=1)(x0, y0)
                grad = np.array([dz_dx, dz_dy]).reshape((2,1))
                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * np.square(grad)
                self.beta1_exp *= self.beta1
                self.beta2_exp *= self.beta2
                w = w - self.eta * (m / (1.0 - self.beta1_exp)) / (np.sqrt(v / (1.0 - self.beta2_exp)) + self.epsilon)
            
            # update 
            x0 = float(w[0])
            y0 = float(w[1])
            x0s += [x0]
            y0s += [y0]
            steps += 1

            # stop at optimal
            if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.01:
                minimum = True

        print (x0, y0, steps)
        return np.array(x0s), np.array(y0s)

    def gradient(self, func, w):
        # calculate gradent
        x0 = float(w[0])
        y0 = float(w[1])
        dz_dx = elementwise_grad(func, argnum=0)(x0, y0)
        dz_dy = elementwise_grad(func, argnum=1)(x0, y0)
        grad = np.array([dz_dx, dz_dy]).reshape((2,1))
        return grad

optimizers = ["SGD", "Momentum", "Nesterov", "Adam", "SGLD", "SGLD-Momentum", "SGLD-Nesterov"]
