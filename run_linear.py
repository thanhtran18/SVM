import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from kernels import *
from SVM import SupportVectorMachine
from utils import *

print('Training linear SVM...')

data=loadmat('data_linear.mat')
X=data['X'].astype(float)
y=np.squeeze(data['y'].astype(int))
y[y == 0]=-1
m=X.shape[0]
hyper_C = 0.01

model=SupportVectorMachine(C=hyper_C, kernel=linear_kernel)
model.fit(X,y)
visualizeBoundaryLinear(X, y, model)
