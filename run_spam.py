from scipy.io import loadmat
from kernels import *
from SVM import SupportVectorMachine
from utils import *

# run_spam.m

# load spam data

data = loadmat('data_spam.mat')
print('\nTraining SVM with custom-defined kernel...\n')
XTrain = data['XTrain']
yTrain = data['yTrain']
XTest = data['XTest']
yTest = data['yTest']
C = 10
# run_spam.m:6
model = linear_kernel(XTrain=XTrain, yTrain=yTrain, C=C, kernel=linear_kernel, sigma=0.001, numberTraining=200)
# run_spam.m:7
pred = SupportVectorMachine.predict(model, XTrain)
# run_spam.m:9
trainAccuracy = ((pred == yTrain) / yTrain.shape[0]).shape[0]
# run_spam.m:10
pred = SupportVectorMachine.predict(model, XTest)
# run_spam.m:12
testAccuracy = ((pred == yTest) / yTest.shape[0]).shape[0]
# run_spam.m:13
dictionary = loadmat('dictionary.mat')
dump, I = -np.sort(-model.w)
# run_spam.m:17
# output the top 10 spam words
print(I(np.arange(1, 10)))
# output the top 10 non-spam words
print(I[-10:])
