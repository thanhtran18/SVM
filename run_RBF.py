import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from kernels import *
from SVM import SupportVectorMachine
from utils import *

data=loadmat('data_RBF.mat')
X=data['X'].astype(float)
y=np.squeeze(data['y'].astype(int))
y[y == 0]=-1
m=X.shape[0]
hyper_C=1
hyper_sigma=0.1

print('Training RBF kernel SVM...')

model=SupportVectorMachine(C=hyper_C, kernel=rbf_kernel, sigma=hyper_sigma)
model.fit(X,y)

print('Visualizing results...')
visualizeBoundary(X, y, model)
