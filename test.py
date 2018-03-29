
from functions import *
f, xmin, xmax, ymin, ymax, minima = TestFunction().get_func(name="McCormick", plot_func=False, plot_gradient=False)

from autograd import numpy as np
from autograd import elementwise_grad, grad
from scipy.linalg import inv, det

import matplotlib.pyplot as plt

def log_likelihood(solution, mu, sigma):
	N = solution.shape[0]
	sigma = sigma * np.eye(2)
	log_likelihood = -(1/2.) * np.log(2*np.pi) - (1/2.) * np.log(det(sigma)) - (1/2.) * np.dot(np.dot((solution-mu).T, inv(sigma)), (solution-mu)) #(solution-mu).T.dot(inv(sigma)).dot(solution-mu)

	return log_likelihood

def log_normal(mu, sigma):
    return -0.5 * (np.log(2*np.pi / sigma) + sigma * (mu**2))

def log_prior_normal(model_params, sigma):
    res = 0.
    for p in model_params:
        res += np.sum(log_normal(p, sigma))
    return res

x0, y0 = -3., 4.

mu = np.array([x0, y0])
sigma = np.array([0.0001,0.001])
popsize = 5000


solutions = np.random.multivariate_normal(mean=mu, cov=sigma*np.eye(2), size=popsize)
plt.plot(x0, y0, "o")
plt.plot(solutions.T[0], solutions.T[1], ".")
plt.show()

rewards = []
for s in solutions:
	rewards += [f(s[0], s[1])]

for s in solutions:
	print log_likelihood(s, mu, sigma)

def log_post(mu, sigma):

	ll = 0
	for i in range(solutions.shape[0]):
		s = solutions[i]
		ll += (log_likelihood(s, mu, sigma) * rewards[i])

	ln = log_prior_normal(mu, sigma)

	# return ln + ll
	return ll

print grad(log_post)(mu, sigma) / popsize


# for s in solutions:
# 	print log_likelihood(s, mu, sigma)


# grad_log_likeihood = np.zeros(2)
# for s in solutions:
# 	# print elementwise_grad(log_likelihood, argnum=1)(s, mu, sigma)
# 	grad_log_likeihood += grad(log_likelihood)(s, mu, sigma) * f(s[0], s[1])
# print grad_log_likeihood/popsize

# print log_prior_normal(mu, sigma)

# grad_log_normal =  grad(log_prior_normal)(mu, sigma)

# print grad_log_normal

# print grad_log_normal + grad_log_likeihood/popsize
