import numpy as np
import ex6_utils
from matplotlib import pyplot
from scipy.io import loadmat


data = loadmat('ex6data1.mat')
X,y = data['X'], data['y'][:,0]


#----------------TRAINING AN SVM-----------------------------------
'Here a function is used which will train a model taking the input parameters, the type of kernel, margin parameter and sigma as arguments.'
'This is an example where a model is trained using just SVM and linear kernel.'

model = ex6_utils.svmTrain(X,y,100,ex6_utils.linearKernel, 1e-3, 20)
ex6_utils.visualizeBoundaryLinear(X,y,model)

#----------------GAUSSIAN KERNEL-----------------------------------
'The model trained by svmTrain function depends on the type of boundary function (aka kernel).'

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
k = ex6_utils.GaussianKernel(x1,x2,2)
print(k)

#----------------LOADING DATASET 2-----------------------------------
'This is an example where a model is trained using SVM with Gaussian kernel'

data2 = loadmat('ex6data2.mat')
X ,y = data2['X'], data2['y'][:,0]
print(X.shape, y.shape)


c2 = 1
sigma = 0.1
model2 = ex6_utils.svmTrain(X,y,c2, ex6_utils.GaussianKernel,args = (sigma,))

#ex6_utils.visualizeBoundary(X,y,model2)
#pyplot.show()

#----------------LOADING DATASET 3-----------------------------------
'In this example, we also have the cross-validation dataset to optimize the values of C and sigma.'

data3 = loadmat('ex6data3.mat')
X, y = data3['X'], data3['y'][:,0]
Xval, yval = data3['Xval'], data3['yval'][:,0]

#ex6_utils.plot(X,y)
#pyplot.show()

c, sigma = ex6_utils.dataset3Params(X,y,Xval,yval)
print(c, sigma)
model3 = ex6_utils.svmTrain(X,y,c,ex6_utils.GaussianKernel,args = (sigma,))
ex6_utils.visualizeBoundary(X,y,model3)
pyplot.show()