import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from kernels import *
from SVM import SupportVectorMachine
from utils import *

print('Training polynomial kernel SVM...')

data=loadmat('data_polynomial.mat')
X=data['X'].astype(float)
y=np.squeeze(data['y'].astype(int))
y[y == 0]=-1
m=X.shape[0]
hyper_C=1
hyper_power=2

model=SupportVectorMachine(C=hyper_C, kernel=polynomial_kernel, power=hyper_power)
model.fit(X,y)

visualizeBoundary(X, y, model)
