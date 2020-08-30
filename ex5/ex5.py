import numpy as np
import os
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import ex5_utils

data = loadmat('ex5data1.mat')

#[:,0] is added after import 'y' data to convert 2-D matrix into numpy vector
X,y = data['X'], data['y'][:,0]
Xtest, ytest = data['Xtest'], data['ytest'][:,0]
Xval, yval = data['Xval'], data['yval'][:,0]

m = y.size

pyplot.plot (X,y,'ro',ms=10,mec='k',mew=1)
pyplot.xlabel('Change in water level(x)')
pyplot.ylabel('Water flowing out of dam(y)')
pyplot.show

'--------------------------------COST FUNCTION-------------------------------------------------'
theta = np.array([1,1])
X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)

J,grad = ex5_utils.costFunc(theta, X_aug, y, 1)
print(J,grad)

'--------------------------------TRAINING LINEAR REGRESSION------------------------------------'
#[This is used to optimize the value of THETA using the FMINCG function.]'''
theta = ex5_utils.trainLinearReg(X_aug,y,0)
print(theta)

pyplot.plot(X,y,'ro', ms=10, mec='k', mew=1.5)
pyplot.plot(X,np.dot(X_aug,theta),'--',lw=2)
pyplot.show()

'--------------------------------GENERATE LEARNING CURVES------------------------------------'
#[This is done to compute the error on the training and the cross validation set and plot the learning curves]
Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
error_train, error_val = ex5_utils.learningCurve(X_aug, y, Xval_aug, yval, lamda=0)

for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

pyplot.plot(np.arange(1,m+1), error_train, np.arange(1,m+1), error_val,lw=2)
pyplot.title('Learning curve for linear regression')
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, 13, 0, 150])
pyplot.show()

'--------------------------------POLYNOMIAL FEATURES------------------------------------'
#[Till now, as per the graph, we saw that our model is highly biased, i.e it is underfitting.
# So in order to minimize underfitting, we will add polynomial features to our training set,
# and eventually to our test and cross-validation set.]

p = 8
X_poly = ex5_utils.polyFeatures(X,p)

'--------------------------------FEATURE NORMALIZE------------------------------------'

X_poly, mu, sigma = ex5_utils.featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m,1)), X_poly], axis=1)

X_poly_test = ex5_utils.polyFeatures(Xtest,p)
X_poly_test = (X_poly_test - mu)/sigma
#X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size,1)), X_poly_test], axis = 1)

X_poly_val = ex5_utils.polyFeatures(Xval,p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.concatenate([np.ones((yval.size,1)), X_poly_val], axis = 1)

print('Normalized Training Examples 1:')
print(X_poly[0,:])

'--------------------------------POLYNOMIAL FIT------------------------------------'

lamda = 0
theta = ex5_utils.trainLinearReg(X_poly,y,lamda)

pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
ex5_utils.polyFit(np.min(X), np.max(X), mu, sigma, theta, p)
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
pyplot.title('Polynomial Regression Fit (lambda = %f)' % lamda)
pyplot.ylim([-20, 50])
pyplot.show()

error_train, error_val = ex5_utils.learningCurve(X_poly, y, X_poly_val, yval, lamda)

pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)
pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lamda)
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, 13, 0, 100])
pyplot.legend(['Train', 'Cross Validation'])
pyplot.show()

print('Polynomial Regression (lambda = %f)\n' % lamda)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))


'--------------------------------VALIDATION SET------------------------------------'
#[The value of λ can significantly affect the results of regularized polynomial regression on
# the training and cross validation set. In particular, a model without regularization (λ = 0)
# fits the training set well, but does not generalize.

# Conversely, a model with too much regularization (λ = 100) does not fit the training set
# and testing set well. A good choice of λ (e.g., λ = 1) can provide a good fit to the data.
# Here we will implement an automated method to select the λ parameter.]

lambda_vec, error_train, error_val = ex5_utils.validationCurve(X_poly, y, X_poly_val, yval)

pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('lambda')
pyplot.ylabel('Error')
pyplot.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

