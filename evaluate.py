from functions import *
from sgd import *




X_opt = []
Y_opt = []
f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="McCormick", plot_func=True, plot_gradient=True)
for optimizer in optimizers:
    print optimizer
    opt = GradientBasedOptimization(optimizer = optimizer)
    x0s, y0s = opt.train(f, 2.5, -1., xmin, xmax, ymin, ymax, minima)
    X_opt += [x0s]
    Y_opt += [y0s]

TestFunction().plot_optimizer(name="McCormick", optimizers=optimizers, tracks=[X_opt, Y_opt])

# X_opt = []
# Y_opt = []
# f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="McCormick", plot_func=False, plot_gradient=False)
# for optimizer in optimizers:
#     print optimizer
#     opt = GradientBasedOptimization(optimizer = optimizer)
#     x0s, y0s = opt.train(f, 1., 1., xmin, xmax, ymin, ymax, minima)
#     X_opt += [x0s]
#     Y_opt += [y0s]

# TestFunction().plot_optimizer(name="McCormick", optimizers=optimizers, tracks=[X_opt, Y_opt])

# X_opt = []
# Y_opt = []
# f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Beale", plot_func=False, plot_gradient=False)
# for optimizer in optimizers:
#     print optimizer
#     opt = GradientBasedOptimization(optimizer = optimizer)
#     x0s, y0s = opt.train(f, 1., -2., xmin, xmax, ymin, ymax, minima)
#     X_opt += [x0s]
#     Y_opt += [y0s]

# TestFunction().plot_optimizer(name="Beale", optimizers=optimizers, tracks=[X_opt, Y_opt])


# X_opt = []
# Y_opt = []
# f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Beale", plot_func=False, plot_gradient=False)
# for optimizer in optimizers:
#     print optimizer
#     opt = GradientBasedOptimization(optimizer = optimizer)
#     x0s, y0s = opt.train(f, 1., 1.5, xmin, xmax, ymin, ymax, minima)
#     X_opt += [x0s]
#     Y_opt += [y0s]

# TestFunction().plot_optimizer(name="Beale", optimizers=optimizers, tracks=[X_opt, Y_opt])

# X_opt = []
# Y_opt = []
# f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Booth", plot_func=True, plot_gradient=True)
# for optimizer in optimizers:
#     print optimizer
#     opt = GradientBasedOptimization(optimizer = optimizer)
#     x0s, y0s = opt.train(f, -8., 1.5, xmin, xmax, ymin, ymax, minima)
#     X_opt += [x0s]
#     Y_opt += [y0s]

# TestFunction().plot_optimizer(name="Booth", optimizers=optimizers, tracks=[X_opt, Y_opt])

# X_opt = []
# Y_opt = []
# f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Booth", plot_func=False, plot_gradient=False)
# for optimizer in optimizers:
#     print optimizer
#     opt = GradientBasedOptimization(optimizer = optimizer)
#     x0s, y0s = opt.train(f, 3., -2.5, xmin, xmax, ymin, ymax, minima)
#     X_opt += [x0s]
#     Y_opt += [y0s]

# TestFunction().plot_optimizer(name="Booth", optimizers=optimizers, tracks=[X_opt, Y_opt])