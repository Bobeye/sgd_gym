"""
Stochastic Gradient Langevin Dynamics

by Bowen, March 2018
"""


from autograd import numpy as np
from autograd import elementwise_grad
<<<<<<< HEAD
from scipy.linalg import inv, det
=======
>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e

class StepSizeGenerator(object):

    def __init__(self, max_epoch, eps_start=0.01, eps_end=0.0001, gamma=0.55):
        self.a, self.b = self.__calc_ab(eps_start, eps_end, gamma, max_epoch)
        self.gamma = gamma

    def __calc_ab(self, eps_start, eps_end, gamma, epoch):
        """Returns coefficients that characterize step size

        Args:
            eps_start(float): initial step size
            eps_end(float): initial step size
            gamma(float): decay rate
            epoch(int): # of epoch
        Returns:
            pair of float: (A, B) satisfies ``A / B ** gamma == eps_start``
            and ``A / (B + epoch) ** gamma == eps_end``
        """

        B = 1 / ((eps_start / eps_end) ** (1 / gamma) - 1) * epoch
        A = eps_start * B ** gamma
        eps_start_actual = A / B ** gamma
        eps_end_actual = A / (B + epoch) ** gamma
        assert abs(eps_start - eps_start_actual) < 1e-4
        assert abs(eps_end - eps_end_actual) < 1e-4
        return A, B

    def __call__(self, epoch):
        return self.a / (self.b + epoch) ** self.gamma

class SGLangevinDynamics():
<<<<<<< HEAD
    var_x0 = 0.5
    var_y0 = 0.5
    sample = 1000
    gamma = 0.55
    max_epoch = 5000
    learning_rate_init = 0.001
    learning_rate_limit = 0.0001
    # a = 10
    # b = 1000
    popsize=200

    def __init__(self, func):
        self.F = func 
        self.ssg = StepSizeGenerator(self.max_epoch, 
                                     eps_start=self.learning_rate_init, 
                                     eps_end=self.learning_rate_limit, 
                                     gamma=self.gamma)
        self.step_size = self.ssg(0)


    def grad_log_normal(self, mu, sigma):
        mu_grad = mu/(sigma**2)
        sigma_grad = (mu**2-sigma**2)/(sigma**3)
        return mu_grad, sigma_grad
=======
    var_x0 = 10.
    var_y0 = 10.
    sample = 10000
    # gamma = 0.55
    # a = 1
    # b = 10000000

    def __init__(self, func, steps_max=10000):
        self.F = func
        self.steps_max = steps_max
        self.ssg = StepSizeGenerator(steps_max, eps_start=0.001, eps_end=0.00001, gamma=0.85) 


    def _log_normal(self, x0, y0):
        # return np.log( np.sqrt(2 * np.pi)) - 1/2 * (self.F(x0, y0)**2)
        # return np.log( np.sqrt(2 * np.pi)) - 1/2 * (x0**2) + np.log( np.sqrt(2 * np.pi)) - 1/2 * (y0**2)
        return np.log( np.sqrt(2 * np.pi) * self.step_var ) - 1 / 2 * (x0**2) / (self.step_var**2) + np.log( np.sqrt(2 * np.pi) * self.step_var ) - 1 / 2 * (y0**2) / (self.step_var**2)

    def _log_likelihood(self, x0, y0):

        xs = x0 + np.random.randn(self.sample) * self.step_var
        ys = y0 + np.random.randn(self.sample) * self.step_var

        return -np.sum((self.F(xs, ys) - self.F(x0, y0))**2)  #/(self.sample)
>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e

    def grad_log_likelihood(self, solution, mu, sigma):
        mu_grad = (solution-mu) / (sigma**2)
        sigma_grad = (((solution-mu) ** 2) - (sigma**2))/(sigma**3)
        return mu_grad, sigma_grad

<<<<<<< HEAD
    def train(self, x0, y0, minima):
        steps_max = self.max_epoch
        x0s = [x0]
        y0s = [y0]
        minimum = False
        step = 1
        self.mu = np.array([x0, y0])
        self.sigma = np.array([self.var_x0, self.var_y0])
        while step<steps_max and not minimum:
            if (self.step_size > self.learning_rate_limit):
                self.step_size = self.ssg(step)
            self.step_var = np.sqrt(self.step_size) 

            
            solutions = [self.mu]
            for _ in range(self.popsize-1):
                solutions += [self.mu + np.random.randn(2) * abs(self.sigma)]
            solutions = np.array(solutions)

            mu_log_normal, sigma_log_normal = self.grad_log_normal(self.mu, self.sigma)
        
            mu_log_likelihood = np.zeros(2)
            sigma_log_likelihood = np.zeros(2)
            rewards = []
            for i in range(self.popsize):
                reward = self.F(solutions[i][0], solutions[i][1])
                rewards += [reward]
            rewards = np.array(rewards)
            # normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            normalized_rewards = (rewards - self.F(self.mu[0], self.mu[1])) / np.max(abs(rewards-self.F(self.mu[0], self.mu[1])))
            for i in range(self.popsize):
                reward = normalized_rewards[i]
                mu_g, sigma_g = self.grad_log_likelihood(solutions[i], self.mu, self.sigma)
                mu_log_likelihood += -reward * mu_g
                sigma_log_likelihood += -reward * sigma_g

            mu_grad = mu_log_normal + mu_log_likelihood/self.popsize
            sigma_grad = sigma_log_normal + sigma_log_likelihood/self.popsize

            delta_mu = ((mu_grad * self.step_size / 2) + np.random.randn(2)*self.step_var)
            delta_sigma = ((sigma_grad * self.step_size / 2) + np.random.randn(2)*self.step_var)
            delta_mu = np.clip(delta_mu, -0.5, 0.5)
            delta_sigma = np.clip(delta_sigma, -0.5, 0.5)            

            self.mu -= delta_mu
            self.sigma -= delta_sigma

            self.sigma = np.sqrt(self.sigma*self.sigma)


            ind = np.argsort(np.array(reward))[0]
            x0 = solutions[ind][0]
            y0 = solutions[ind][1]
=======
    def gradient(self, x0, y0):
        # calculate gradent
        # f = lambda x, y: self._log_gradient(x, y)
        # dz_dx = elementwise_grad(f, argnum=0)(x0, y0)
        # dz_dy = elementwise_grad(f, argnum=1)(x0, y0)
        # grad = np.array([dz_dx, dz_dy]).reshape((2,1))


        f1 = lambda x, y: self._log_normal(x, y)
        f2 = lambda x, y: self._log_likelihood(x, y)
        dz_dx1 = elementwise_grad(f1, argnum=0)(x0, y0)
        dz_dy1 = elementwise_grad(f1, argnum=1)(x0, y0)
        dz_dx2 = elementwise_grad(f2, argnum=0)(x0, y0) / self.sample
        dz_dy2 = elementwise_grad(f2, argnum=1)(x0, y0) / self.sample
        grad = np.array([dz_dx1 + dz_dx2, dz_dy1 + dz_dy2]).reshape((2,1))

        return grad

    def train(self, x0, y0, minima):
        x0s = [x0]
        y0s = [y0]
        minimum = False
        step = 0
        w = np.matrix(np.array([x0, y0]).reshape((2,1)))
        while step<self.steps_max and not minimum:
            # self.step_size = self.a * (self.b + step) ** (-self.gamma)
            self.step_size = self.ssg(step)
            self.step_var = np.sqrt(self.step_size) 

            dw = (self.step_size / 2) *self.gradient(x0, y0) + np.random.randn()*self.step_var
            w = w + dw 
            x0 = float(w[0])
            y0 = float(w[1])
>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e
            x0s += [x0]
            y0s += [y0]

            # stop at optimal
            if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.01:
                minimum = True

<<<<<<< HEAD
            print (x0, y0, mu_grad, sigma_grad, self.step_var, step)
=======
            print (x0, y0, self.step_var, dw, step)
>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e

            step += 1

        return np.array(x0s), np.array(y0s)

if __name__ == "__main__":
    from functions import *
<<<<<<< HEAD
    f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Booth", plot_func=False, plot_gradient=False)
=======
    f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="McCormick", plot_func=False, plot_gradient=False)
>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e
    X_opt = []
    Y_opt = []
    optimizers = ["SGLD"]
    for optimizer in optimizers:
        print optimizer
<<<<<<< HEAD
        opt = SGLangevinDynamics(f)
        x0s, y0s = opt.train(-5., -2., minima)
=======
        opt = SGLangevinDynamics(f, steps_max=10000)
        x0s, y0s = opt.train(0., 2., minima)
>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e
        # x0s, y0s = opt.train(1., 2.)
        X_opt += [x0s]
        Y_opt += [y0s]

<<<<<<< HEAD
    TestFunction().plot_optimizer(name="Booth", optimizers=optimizers, tracks=[X_opt, Y_opt])
=======
    TestFunction().plot_optimizer(name="McCormick", optimizers=optimizers, tracks=[X_opt, Y_opt])



# class StepSizeGenerator(object):

#     def __init__(self, max_epoch, eps_start=0.5, eps_end=0.01, gamma=0.55):
#         self.a, self.b = self.__calc_ab(eps_start, eps_end, gamma, max_epoch)
#         self.gamma = gamma

#     def __calc_ab(self, eps_start, eps_end, gamma, epoch):
#         """Returns coefficients that characterize step size

#         Args:
#             eps_start(float): initial step size
#             eps_end(float): initial step size
#             gamma(float): decay rate
#             epoch(int): # of epoch
#         Returns:
#             pair of float: (A, B) satisfies ``A / B ** gamma == eps_start``
#             and ``A / (B + epoch) ** gamma == eps_end``
#         """

#         B = 1 / ((eps_start / eps_end) ** (1 / gamma) - 1) * epoch
#         A = eps_start * B ** gamma
#         eps_start_actual = A / B ** gamma
#         eps_end_actual = A / (B + epoch) ** gamma
#         assert abs(eps_start - eps_start_actual) < 1e-4
#         assert abs(eps_end - eps_end_actual) < 1e-4
#         return A, B

#     def __call__(self, epoch):
#         return self.a / (self.b + epoch) ** self.gamma



# class StochasticGradientLangevinDynamics():
#     sample = 100

#     def __init__(self, func, xmin, xmax, ymin, ymax):
#         self.F = func
#         self.VAR_1 = ((xmax-xmin) / 100.)**2
#         self.VAR_2 = ((ymax-ymin) / 100.)**2
#         self.VAR_X = 10.

#     def gaussian_likelihood(self, x, mu, var):
#         """
#         Returns likelihood of ``x``, or ``N(x; mu, var)``
#         """
#         return np.exp(-(x - mu) ** 2 / var / 2) / np.sqrt(2 * np.pi * var)

#     def log_likelihood(self, x0, y0):
#         theta1, theta2 = self.theta1, self.theta2

#         # prior
#         log_prior1 = -0.5 * (np.log(2*np.pi / 1.) + 1. * x0**2)
#         log_prior2 = -0.5 * (np.log(2*np.pi / 1.) + 1. * y0**2)
#         # log_prior1 = np.sum(np.log(self.gaussian_likelihood(x0, 0, self.ets)))
#         # log_prior2 = np.sum(np.log(self.gaussian_likelihood(y0, 0, self.ets)))
#         # log_prior1 = x0
#         # log_prior2 = y0
#         # likelihood
#         xs = x0 + np.random.randn(self.sample) * np.sqrt(self.ets)
#         ys = y0 + np.random.randn(self.sample) * np.sqrt(self.ets)
#         log_likelihood = -np.sum((self.F(xs, ys) - self.F(theta1, theta2))**2)/self.sample
        
#         return log_prior1 + log_prior2 + log_likelihood


#         # xs = x0 + np.random.randn(self.sample) * np.sqrt(self.VAR_1)
#         # ys = x0 + np.random.randn(self.sample) * np.sqrt(self.VAR_2)
#         # log_prob =  np.sum(np.log(self.gaussian_likelihood(self.F(xs, ys), self.F(theta1, theta2), self.VAR_X))/self.sample)

#         # return log_prior1 + log_prior2 + log_prob

#         # theta1, theta2 = x0, y0
#         # xs = x0 + np.random.randn(self.sample) * np.sqrt(self.VAR_1)
#         # ys = x0 + np.random.randn(self.sample) * np.sqrt(self.VAR_2)
#         # log_prior1 = np.sum(np.log(self.gaussian_likelihood(theta1, 0, self.VAR_1)))
#         # log_prior2 = np.sum(np.log(self.gaussian_likelihood(theta2, 0, self.VAR_2)))
#         # prob = self.gaussian_likelihood(self.F(xs, ys), self.F(theta1, theta2), self.VAR_X)
#         # # prob2 = self.gaussian_likelihood(self.F(xs, ys), theta2, self.VAR_X)
#         # # log_prob = np.sum(np.log(prob1 / 2. + prob2 / 2.)/self.sample)
#         # log_prob = np.sum(np.log(prob)/self.sample)
#         # return log_prior1 + log_prior2 + log_prob

#     def gradient(self, x0, y0):
#         # calculate gradent
#         f = lambda x, y: self.log_likelihood(x, y)
#         dz_dx = elementwise_grad(f, argnum=0)(x0, y0)
#         dz_dy = elementwise_grad(f, argnum=1)(x0, y0)
#         grad = np.array([dz_dx, dz_dy]).reshape((2,1))
#         return grad

#     def train(self, x0, y0, minima):
#         x0s = []
#         y0s = []
#         minimum = False
#         self.steps = 0
#         ssg = StepSizeGenerator(10000, 0.001, 0.0001)
#         self.ets = ssg(self.steps)
#         w = np.matrix(np.array([x0, y0]).reshape((2,1)))
#         self.theta1, self.theta2 = float(w[0]), float(w[1])
#         while x0 >= xmin and x0 <= xmax and y0 >= ymin and y0 <= ymax and not minimum and self.steps<10000:
#             d_w = self.gradient(x0, y0)
#             self.ets = ssg(self.steps)
#             eta = np.random.randn() * np.sqrt(ssg(self.ets))
#             # w = w + (d_w * self.ets / 2 + eta)
#             w = w + 0.5 * 0.000001 * d_w + np.random.randn() * np.sqrt(1.)

#             self.theta1, self.theta2 = float(w[0]), float(w[1])
#             x0 = float(w[0])
#             y0 = float(w[1])
#             x0s += [x0]
#             y0s += [y0]
#             self.steps += 1

#             # stop at optimal
#             if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.01:
#                 minimum = True

#             # print self.ets
#             print (x0, y0, self.steps)

#         return np.array(x0s), np.array(y0s)

# if __name__ == "__main__":
#     from functions import *
#     f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Beale", plot_func=False, plot_gradient=False)
#     X_opt = []
#     Y_opt = []
#     optimizers = ["SGLD"]
#     for optimizer in optimizers:
#         print optimizer
#         opt = StochasticGradientLangevinDynamics(f, xmin, xmax, ymin, ymax)
#         x0s, y0s = opt.train(1., 2., minima)
#         X_opt += [x0s]
#         Y_opt += [y0s]

#     TestFunction().plot_optimizer(name="Beale", optimizers=optimizers, tracks=[X_opt, Y_opt])












# class GradientBasedOptimization():

#     def __init__(self, optimizer=None):
#         self.Lambda = 0.01

#         self.gamma = 0.9
#         self.epsilon = 1e-8

#         # Adam
#         self.beta1 = 0.9
#         self.beta2 = 0.999
#         self.beta1_exp = 1.0
#         self.beta2_exp = 1.0
        
#         # Adadelta & RMSprop params
#         self.grad_expect = np.matrix(np.zeros((2,1)))
#         self.delta_expect = np.matrix(np.zeros((2,1)))

#         if optimizer == 'Adam':
#             self.eta = 0.05
#         elif optimizer == 'RMSprop':
#             self.eta = 0.05
#         elif optimizer == "Adagrad":
#             self.eta = 0.3
#         elif optimizer == "Adadelta":
#             self.eta = 0.0001
#         else:
#             self.eta = 0.001

#         self.optimizer = optimizer


#     def train(self, func, x0, y0, xmin, xmax, ymin, ymax, minima):
#         x0s = [x0]
#         y0s = [y0]

#         minimum = False
#         init = True
#         steps = 0

#         w = np.matrix(np.array([x0, y0]).reshape((2,1)))
#         v = np.matrix(np.zeros(w.shape))
#         m = np.matrix(np.zeros(w.shape))
#         grad_sum_square = np.matrix(np.zeros(w.shape))    # for adagrad

#         while x0 >= xmin and x0 <= xmax and y0 >= ymin and y0 <= ymax and not minimum and steps<20000:

#             if self.optimizer == 'SGD':
#                 # Stochastic gradient descent
#                 grad = self.gradient(func, w)
#                 w = w - self.eta * grad
#             elif self.optimizer == 'Momentum':
#                 # Momentum
#                 grad = self.gradient(func, w)
#                 v = self.gamma * v + self.eta * grad
#                 w = w - v
#             elif self.optimizer == 'Nesterov':
#                 # Nesterov accelerated gradient
#                 w = w - self.gamma * v
#                 grad = self.gradient(func, w)
#                 v = self.gamma * v + self.eta * grad
#                 w = w - v
#             elif self.optimizer == 'Adagrad':
#                 # Adagrad
#                 grad = self.gradient(func, w)
#                 grad_sum_square += np.square(grad)
#                 w = w - ((self.eta * grad) / (np.sqrt(grad_sum_square)+ self.epsilon))
#             elif self.optimizer == 'Adadelta':
#                 # Adadelta
#                 grad = self.gradient(func, w)
#                 self.grad_expect = self.gamma * self.grad_expect + (1.0 - self.gamma) * np.square(grad)
#                 if init == True:    # use sgd
#                     delta = - self.eta * grad 
#                     init = False
#                 else:    
#                     delta = - np.multiply(np.sqrt(self.delta_expect + self.epsilon) / np.sqrt(self.grad_expect + self.epsilon),  grad)
#                 w = w + delta
#                 self.delta_expect = self.gamma * self.delta_expect + (1.0 - self.gamma) * np.square(delta)
#             elif self.optimizer == 'RMSprop':
#                 # RMSprop
#                 grad = self.gradient(func, w)
#                 self.grad_expect = self.gamma * self.grad_expect + (1.0 - self.gamma) * np.square(grad)
#                 w = w - self.eta * grad / np.sqrt(self.grad_expect + self.epsilon)
#             elif self.optimizer == 'Adam':
#                 # Adam
#                 grad = self.gradient(func, w)
#                 dz_dx = elementwise_grad(func, argnum=0)(x0, y0)
#                 dz_dy = elementwise_grad(func, argnum=1)(x0, y0)
#                 grad = np.array([dz_dx, dz_dy]).reshape((2,1))
#                 m = self.beta1 * m + (1.0 - self.beta1) * grad
#                 v = self.beta2 * v + (1.0 - self.beta2) * np.square(grad)
#                 self.beta1_exp *= self.beta1
#                 self.beta2_exp *= self.beta2
#                 w = w - self.eta * (m / (1.0 - self.beta1_exp)) / (np.sqrt(v / (1.0 - self.beta2_exp)) + self.epsilon)
            



#             # update 
#             x0 = float(w[0])
#             y0 = float(w[1])
#             x0s += [x0]
#             y0s += [y0]
#             steps += 1

#             # stop at optimal
#             if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.01:
#                 minimum = True

#         print (x0, y0, steps)
#         return np.array(x0s), np.array(y0s)

#     def gradient(self, func, w):
#         # calculate gradent
#         x0 = float(w[0])
#         y0 = float(w[1])
#         dz_dx = elementwise_grad(func, argnum=0)(x0, y0)
#         dz_dy = elementwise_grad(func, argnum=1)(x0, y0)
#         grad = np.array([dz_dx, dz_dy]).reshape((2,1))
#         return grad


>>>>>>> 83e343a8189922328e10752cb9195c59c58c943e

