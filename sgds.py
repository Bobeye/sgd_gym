import numpy as np
import autograd
from autograd import elementwise_grad


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation



"""
Test functions for optimization:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
class TestFunction():

    def __init__(self):
        pass

    def init_beale(self, step=0.2):
        self.f = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        self.xmin, self.xmax = -4.5, 4.5
        self.ymin, self.ymax = -4.5, 4.5
        self.xs, self.ys = np.meshgrid(np.arange(self.xmin, self.xmax + step, step), np.arange(self.ymin, self.ymax + step, step))
        self.zs = self.f(self.xs, self.ys)
        self.minima = np.array([3., .5])

    def init_booth(self, step=0.2):
        self.f = lambda x, y: (x + 2*y - 7) ** 2 + (2*x +y -5) ** 2
        self.xmin, self.xmax = -10., 10.
        self.ymin, self.ymax = -10., 10.
        self.xs, self.ys = np.meshgrid(np.arange(self.xmin, self.xmax + step, step), np.arange(self.ymin, self.ymax + step, step))
        self.zs = self.f(self.xs, self.ys)
        self.minima = np.array([1., 3.])

    def get_func(self, name="Beale", plot_func=False, plot_gradient=False):
        if name == "Beale":
            self.init_beale()
        elif name == "Booth":
            self.init_booth()
            
        if plot_func:
            fig = plt.figure(figsize=(8, 5))
            ax = plt.axes(projection='3d', elev=50, azim=-50)
            ax.plot_surface(self.xs, self.ys, self.zs, norm=LogNorm(), rstride=1, cstride=1, 
                            edgecolor='none', alpha=.8, cmap=plt.cm.jet)
            minima_ = self.minima.reshape(-1, 1)
            ax.plot(minima_[0], minima_[1], self.f(*minima_), 'r*', markersize=10)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')

            ax.set_xlim((self.xmin, self.xmax))
            ax.set_ylim((self.ymin, self.ymax))

            plt.show()

        if plot_gradient:
            dz_dx = elementwise_grad(self.f, argnum=0)(self.xs, self.ys)
            dz_dy = elementwise_grad(self.f, argnum=1)(self.xs, self.ys)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contour(self.xs, self.ys, self.zs, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
            ax.quiver(self.xs, self.ys, self.xs - dz_dx, self.ys - dz_dy, alpha=.5)
            ax.plot(self.minima[0], self.minima[1], 'r*', markersize=18)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')

            ax.set_xlim((self.xmin, self.xmax))
            ax.set_ylim((self.ymin, self.ymax))

            plt.show()

        return self.f, self.xmin, self.xmax, self.ymin, self.ymax, self.minima

    def plot_optimizer(self, name="Beale", optimizers=None, tracks=None):
        if name == "Beale":
            self.init_beale()
        elif name == "Booth":
            self.init_booth()

        dz_dx = elementwise_grad(self.f, argnum=0)(self.xs, self.ys)
        dz_dy = elementwise_grad(self.f, argnum=1)(self.xs, self.ys)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.xs, self.ys, self.zs, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.quiver(self.xs, self.ys, self.xs - dz_dx, self.ys - dz_dy, alpha=.5)
        ax.plot(self.minima[0], self.minima[1], 'r*', markersize=18)

        for i in range(len(optimizers)):
            ax.plot(tracks[0][i], tracks[1][i], "-", label=optimizers[i]+"_"+str(tracks[0][i].shape[0]))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))

        plt.legend()
        plt.show()




"""
SGD
"""
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
        elif optimizer == 'RMSprop':
            self.eta = 0.05
        elif optimizer == "Adagrad":
            self.eta = 0.3
        elif optimizer == "Adadelta":
            self.eta = 0.0001
        else:
            self.eta = 0.001

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

        while x0 >= xmin and x0 <= xmax and y0 >= ymin and y0 <= ymax and not minimum and steps<=20000:

            if self.optimizer == 'SGD':
                # Stochastic gradient descent
                grad = self.gradient(func, w)
                w = w - self.eta * grad
            elif self.optimizer == 'Momentum':
                # Momentum
                grad = self.gradient(func, w)
                v = self.gamma * v + self.eta * grad
                w = w - v
            elif self.optimizer == 'Nesterov':
                # Nesterov accelerated gradient
                w = w - self.gamma * v
                grad = self.gradient(func, w)
                v = self.gamma * v + self.eta * grad
                w = w - v
            elif self.optimizer == 'Adagrad':
                # Adagrad
                grad = self.gradient(func, w)
                grad_sum_square += np.square(grad)
                w = w - ((self.eta * grad) / (np.sqrt(grad_sum_square)+ self.epsilon))
            elif self.optimizer == 'Adadelta':
                # Adadelta
                grad = self.gradient(func, w)
                self.grad_expect = self.gamma * self.grad_expect + (1.0 - self.gamma) * np.square(grad)
                if init == True:    # use sgd
                    delta = - self.eta * grad 
                    init = False
                else:    
                    delta = - np.multiply(np.sqrt(self.delta_expect + self.epsilon) / np.sqrt(self.grad_expect + self.epsilon),  grad)
                w = w + delta
                self.delta_expect = self.gamma * self.delta_expect + (1.0 - self.gamma) * np.square(delta)
            elif self.optimizer == 'RMSprop':
                # RMSprop
                grad = self.gradient(func, w)
                self.grad_expect = self.gamma * self.grad_expect + (1.0 - self.gamma) * np.square(grad)
                w = w - self.eta * grad / np.sqrt(self.grad_expect + self.epsilon)
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
            if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.0001:
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

optimizers = ["SGD", "Momentum", "Nesterov", "Adagrad", "Adadelta", "RMSprop", "Adam"]


if __name__ == "__main__":
    
    # X_opt = []
    # Y_opt = []
    # f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Booth", plot_func=False, plot_gradient=False)
    # for optimizer in optimizers:
    #     print optimizer
    #     opt = GradientBasedOptimization(optimizer = optimizer)
    #     x0s, y0s = opt.train(f, -5., -7., xmin, xmax, ymin, ymax, minima)
    #     X_opt += [x0s]
    #     Y_opt += [y0s]

    # TestFunction().plot_optimizer(name="Booth", optimizers=optimizers, tracks=[X_opt, Y_opt])

    X_opt = []
    Y_opt = []
    f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Beale", plot_func=False, plot_gradient=False)
    for optimizer in optimizers:
        print optimizer
        opt = GradientBasedOptimization(optimizer = optimizer)
        x0s, y0s = opt.train(f, 1., 1., xmin, xmax, ymin, ymax, minima)
        X_opt += [x0s]
        Y_opt += [y0s]

    TestFunction().plot_optimizer(name="Beale", optimizers=optimizers, tracks=[X_opt, Y_opt])



    

    