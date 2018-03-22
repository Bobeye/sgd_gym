from functions import *
from sgd import *


X_opt = []
Y_opt = []
f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="McCormick", plot_func=True, plot_gradient=False)
for optimizer in optimizers:
    print optimizer
    opt = GradientBasedOptimization(optimizer = optimizer)
    x0s, y0s = opt.train(f, 3., 3., xmin, xmax, ymin, ymax, minima)
    X_opt += [x0s]
    Y_opt += [y0s]

TestFunction().plot_optimizer(name="McCormick", optimizers=optimizers, tracks=[X_opt, Y_opt])