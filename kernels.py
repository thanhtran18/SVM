import numpy as np


def linear_kernel(**kwargs):
    def f(x1, x2):
        # fill in your code for linear kernel
        # Make x1 and x2 are column vectors
        x1 = x1.reshape(x1.shape[0], 1)
        x2 = x2.reshape(x2.shape[0], 1)
        # Compute the kernel
        kernel = np.dot(x1.T, x2)
        return kernel[0, 0]

    return f


def polynomial_kernel(power, **kwargs):
    def f(x1, x2):
        # fill in your code for polynomial kernel (with hyparameter "power" as the degree of polynomial)
        x1 = x1.reshape(x1.shape[0], 1)
        x2 = x2.reshape(x2.shape[0], 1)
        kernel = (1 + np.dot(x1.T, x2)) ** power
        return kernel
    return f
#


def rbf_kernel(sigma, **kwargs):
    def f(x1, x2):
        # fill in your code for RBF kernel (with hyparameter "sigma")
        x1 = np.ravel(x1).reshape(x1.shape[0], 1)
        x2 = np.ravel(x2).reshape(x2.shape[0], 1)
        K = np.exp(- sum((x1 - x2) ** 2) / (np.dot(2, sigma ** 2)))

    return f
