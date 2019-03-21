import numpy as np


def linear_kernel(**kwargs):
    def f(x1, x2):
        # fill in your code for linear kernel
        # Ensure that x1 and x2 are column vectors
        x1 = np.ravel(x1).reshape(x1.shape[0], 1)
        # x1 = x1[:].reshape(x1.shape[0], 1)
        print(x1)

        # linearKernel.m:7
        x2 = np.ravel(x2).reshape(x2.shape[0], 1)
        # x2 = x2[:].reshape(x2.shape[0], 1)
        print(x2)
        # # linearKernel.m:7
        # # Compute the kernel
        K = np.dot(x1.T, x2)

    return f
#
#
# def polynomial_kernel(power, **kwargs):
#     def f(x1, x2):
#         x1 = np.ravel(x1).reshape(x1.shape[0], 1)
#         # polynomialKernel.m:7
#         x2 = np.ravel(x2).reshape(x2.shape[0], 1)
#         # polynomialKernel.m:7
#         # Compute the kernel
#         K = (1 + np.dot(x1.T, x2)) ** power
#         # fill in your code for polynomial kernel (with hyparameter "power" as the degree of polynomial)
#
#     return f
# #
# #
def rbf_kernel(sigma, **kwargs):
    def f(x1, x2):
        # fill in your code for RBF kernel (with hyparameter "sigma")
        x1 = np.ravel(x1).reshape(x1.shape[0], 1)
        # RBFKernel.m:7
        x2 = np.ravel(x2).reshape(x2.shape[0], 1)
        # RBFKernel.m:7
        K = np.exp(- sum((x1 - x2) ** 2) / (np.dot(2, sigma ** 2)))

    return f
