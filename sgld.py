"""
Stochastic Gradient Langevin Dynamics

by Bowen, March 2018
"""


from autograd import numpy as np
from autograd import elementwise_grad
from scipy.linalg import inv, det

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

    def grad_log_likelihood(self, solution, mu, sigma):
        mu_grad = (solution-mu) / (sigma**2)
        sigma_grad = (((solution-mu) ** 2) - (sigma**2))/(sigma**3)
        return mu_grad, sigma_grad

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
            x0s += [x0]
            y0s += [y0]

            # stop at optimal
            if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.01:
                minimum = True

            print (x0, y0, mu_grad, sigma_grad, self.step_var, step)

            step += 1

        return np.array(x0s), np.array(y0s)

if __name__ == "__main__":
    from functions import *
    f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="Booth", plot_func=False, plot_gradient=False)
    X_opt = []
    Y_opt = []
    optimizers = ["SGLD"]
    for optimizer in optimizers:
        print optimizer
        opt = SGLangevinDynamics(f)
        x0s, y0s = opt.train(-5., -2., minima)
        # x0s, y0s = opt.train(1., 2.)
        X_opt += [x0s]
        Y_opt += [y0s]

    TestFunction().plot_optimizer(name="Booth", optimizers=optimizers, tracks=[X_opt, Y_opt])

