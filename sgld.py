"""
Stochastic Gradient Langevin Dynamics

by Bowen, March 2018
"""


from autograd import numpy as np
from autograd import elementwise_grad, grad
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
    var_x0 = 0.3
    var_y0 = 0.3
    sample = 1000
    gamma = 0.55
    max_epoch = 2000
    learning_rate_init = 0.1
    learning_rate_limit = 0.001
    # a = 10
    # b = 1000
    popsize=10

    def __init__(self, func):
        self.F = func 
        self.ssg = StepSizeGenerator(self.max_epoch, 
                                     eps_start=self.learning_rate_init, 
                                     eps_end=self.learning_rate_limit, 
                                     gamma=self.gamma)
        self.step_size = self.ssg(0)


    # def grad_log_normal(self, mu, sigma):
    #     # mu_grad = mu/(sigma**2)
    #     # sigma_grad = (mu**2-sigma**2)/(sigma**3)
    #     # return mu_grad, sigma_grad



    #     return np.zeros(2), np.zeros(2)

    # def grad_log_likelihood(self, solution, mu, sigma):

    #     mu_grad = (solution-mu) / (sigma**2)
    #     sigma_grad = (((solution-mu) ** 2) - (sigma**2))/(sigma**3)
    #     return mu_grad, sigma_grad

    def log_likelihood(self, solution, mu, sigma):
        N = solution.shape[0]
        sigma = sigma * np.eye(2)
        log_likelihood = -(1/2.) * np.log(2*np.pi) - (1/2.) * np.log(det(sigma)) - (1/2.) * np.dot(np.dot((solution-mu).T, inv(sigma)), (solution-mu)) #(solution-mu).T.dot(inv(sigma)).dot(solution-mu)

        return log_likelihood

    def log_normal(self, mu, sigma):
        sigma = sigma * np.eye(2)
        return -(1/2.) * np.log(2*np.pi) - (1/2.) * np.log(det(sigma)) - (1/2.) * np.dot(np.dot((mu).T, inv(sigma)), (mu))

    def log_prior_normal(self, mu, sigma):
        res = 0.
        for p in mu:
            res += np.sum(self.log_normal(p, sigma))
        return res

    def log_post(self, mu, sigma):

        ll = 0
        for i in range(self.popsize):
            s = self.solutions[i]
            ll += (self.log_likelihood(s, mu, sigma) * self.rewards[i])

        ln = self.log_prior_normal(mu, sigma)

        return ln + ll/self.popsize
        # return ll

    # def sum_log_likelihood(self, mu, sigma):

    #     solutions = [mu]
    #     for _ in range(self.popsize-1):
    #         solutions += [mu + np.random.randn(2) * abs(sigma)]
    #     solutions = np.array(solutions)

    #     sll = 0
    #     for solution in solutions:
    #         x = self.F(solution[0], solution[1])
    #         m = self.F(mu[0], mu[1])
    #         sll += (x-m)**2

    #     return sll / self.popsize



    def train(self, x0, y0, minima):
        steps_max = self.max_epoch
        x0s = [x0]
        y0s = [y0]
        xs = []
        ys = []
        minimum = False
        step = 1
        self.mu = np.array([x0, y0])
        self.sigma = np.array([self.var_x0, self.var_y0])

        self.best_mu = None
        self.best_reward = None
        
        # for Adam
        self.mu_v = np.zeros(2)
        self.sigma_v = np.zeros(2)
        self.mu_m = np.zeros(2)
        self.sigma_m = np.zeros(2)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1_exp = 1.0
        self.beta2_exp = 1.0
        self.eta = 0.05
        self.epsilon = 1e-8
        
        try:
            while step<steps_max and not minimum:
                if (self.step_size > self.learning_rate_limit):
                    self.step_size = self.ssg(step)
                # self.step_size = 0.001
                self.step_var = np.sqrt(self.step_size) 

                
                # solutions = [self.mu]
                # for _ in range(self.popsize-1):
                #     solutions += [self.mu + np.random.randn(2) * abs(self.sigma)]
                # self.solutions = np.array(solutions)
                solutions = np.random.multivariate_normal(mean=self.mu, cov=self.sigma*np.eye(2), size=self.popsize)
                self.solutions = np.array(solutions)
                for s in solutions:
                    xs += [s[0]]
                    ys += [s[1]]
                # mu_log_normal, sigma_log_normal = self.grad_log_normal(self.mu, self.sigma)
            
                # mu_log_likelihood = np.zeros(2)
                # sigma_log_likelihood = np.zeros(2)
                

                # grad_log_likelihood = lambda m, s: self.sum_log_likelihood(m, s)
                # mu_log_likelihood = elementwise_grad(grad_log_likelihood, argnum=0)(self.mu, self.sigma)
                # sigma_log_likelihood = elementwise_grad(grad_log_likelihood, argnum=1)(self.mu, self.sigma)


                # print (mu_log_normal, sigma_log_normal, mu_log_likelihood, sigma_log_likelihood)

                rewards = []
                for i in range(self.popsize):
                    reward = self.F(solutions[i][0], solutions[i][1])
                    rewards += [reward]
                rewards = np.array(rewards)
                self.rewards = (rewards - np.mean(rewards)) / np.std(rewards)


                gradient = grad(self.log_post)(self.mu, self.sigma) / self.popsize
                mu_grad, sigma_grad = gradient[0], gradient[1]
                # normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                # rewards_update = rewards - self.F(self.mu[0], self.mu[1])
                # normalized_rewards = (rewards_update-np.mean(rewards_update)) / np.std(rewards_update)


                # for i in range(self.popsize):
                #     reward = normalized_rewards[i]
                #     mu_g, sigma_g = self.grad_log_likelihood(solutions[i], self.mu, self.sigma)
                #     mu_log_likelihood += reward * mu_g
                #     sigma_log_likelihood += reward * sigma_g



                # mu_grad = mu_log_normal + mu_log_likelihood/self.popsize
                # sigma_grad = sigma_log_normal + sigma_log_likelihood/self.popsize

                delta_mu, delta_sigma = mu_grad, sigma_grad

                # delta_mu = ((mu_grad * self.step_size / 2) + np.random.randn(2)*self.step_var)
                # delta_sigma = ((sigma_grad * self.step_size / 2) + np.random.randn(2)*self.step_var)
                # delta_mu = np.clip(delta_mu, -0.5, 0.5)
                # delta_sigma = np.clip(delta_sigma, -0.5, 0.5)            

                self.mu -= delta_mu * self.step_size
                self.sigma -= delta_sigma * self.step_size



                # self.mu_m = self.beta1 * self.mu_m + (1.0 - self.beta1) * delta_mu
                # self.mu_v = self.beta2 * self.mu_v + (1.0 - self.beta2) * np.square(delta_mu)
                # self.beta1_exp *= self.beta1
                # self.beta2_exp *= self.beta2
                # self.mu -= self.eta * (self.mu_m / (1.0 - self.beta1_exp)) / (np.sqrt(self.mu_v / (1.0 - self.beta2_exp)) + self.epsilon)

                # self.sigma_m = self.beta1 * self.sigma_m + (1.0 - self.beta1) * delta_sigma
                # self.sigma_v = self.beta2 * self.sigma_v + (1.0 - self.beta2) * np.square(delta_sigma)
                # self.beta1_exp *= self.beta1
                # self.beta2_exp *= self.beta2
                # self.sigma -= self.eta * (self.sigma_m / (1.0 - self.beta1_exp)) / (np.sqrt(self.sigma_v / (1.0 - self.beta2_exp)) + self.epsilon)

                
                # # self.mu -= delta_mu
                # # self.sigma -= delta_sigma

                # self.sigma = np.sqrt(self.sigma*self.sigma)


                ind = np.argsort(np.array(rewards))[0]
                x0 = solutions[ind][0]
                y0 = solutions[ind][1]


                if self.best_mu is None or self.best_reward >= rewards[ind]:
                    self.best_mu = [x0, y0]
                    self.best_reward = rewards[ind]
                    x0s += [x0]
                    y0s += [y0]
                else:
                    x0s += [self.best_mu[0]]
                    y0s += [self.best_mu[1]]

                print self.best_reward

                # x0s += [x0]
                # y0s += [y0]

                # stop at optimal
                if (x0-minima[0])**2 + (y0-minima[1])**2 <= 0.3:
                    minimum = True

                print (x0, y0, self.mu, self.sigma, self.step_var, step)

                step += 1
        except KeyboardInterrupt:
            pass

        return np.array(x0s), np.array(y0s), np.array(xs), np.array(ys)

if __name__ == "__main__":
    from functions import *
    f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="McCormick", plot_func=False, plot_gradient=False)
    X_opt = []
    Y_opt = []
    X_sample = []
    Y_sample = []
    optimizers = ["SGLD"]
    for optimizer in optimizers:
        print optimizer
        opt = SGLangevinDynamics(f)
        x0s, y0s, xs, ys = opt.train(0., 2., minima)
        # x0s, y0s = opt.train(1., 2.)
        print (x0s, y0s)
        X_opt += [x0s]
        Y_opt += [y0s]
        X_sample += [xs]
        Y_sample += [ys]

    TestFunction().plot_optimizer(name="McCormick", optimizers=optimizers, tracks=[X_opt, Y_opt], samples=[X_sample, Y_sample])


