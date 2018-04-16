"""
Test functions for optimization

For more information: https://en.wikipedia.org/wiki/Test_functions_for_optimization

by Bowen, March 2018
"""



from autograd import numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm



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

    def init_mccormick(self, step=0.2):
        self.f = lambda x, y: np.sin(x+y) + (x-y) ** 2 - 1.5 * x + 2.5 * y + 1
        self.xmin, self.xmax = -1.5, 4.
        self.ymin, self.ymax = -3., 4.
        self.xs, self.ys = np.meshgrid(np.arange(self.xmin, self.xmax + step, step), np.arange(self.ymin, self.ymax + step, step))
        self.zs = self.f(self.xs, self.ys)
        self.minima = np.array([-0.54719, -1.54719])

    def get_func(self, name="Beale", plot_func=False, plot_gradient=False):
        if name == "Beale":
            self.init_beale()
        elif name == "Booth":
            self.init_booth()
        elif name == "McCormick":
            self.init_mccormick()
            
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

    def plot_optimizer(self, name="Beale", optimizers=None, tracks=None, samples=None):
        if name == "Beale":
            self.init_beale()
        elif name == "Booth":
            self.init_booth()
        elif name == "McCormick":
            self.init_mccormick()

        dz_dx = elementwise_grad(self.f, argnum=0)(self.xs, self.ys)
        dz_dy = elementwise_grad(self.f, argnum=1)(self.xs, self.ys)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.xs, self.ys, self.zs, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.quiver(self.xs, self.ys, self.xs - dz_dx, self.ys - dz_dy, alpha=.5)
        ax.plot(self.minima[0], self.minima[1], 'r*', markersize=18)

        for i in range(len(optimizers)):
            ax.plot(tracks[0][i], tracks[1][i], "-", linewidth=2.0, label=optimizers[i]+":"+str(tracks[0][i].shape[0]-1)+" steps")
            if samples is not None:
                ax.plot(samples[0][i], samples[1][i], ".", alpha=0.3)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))

        plt.legend()
        plt.show()