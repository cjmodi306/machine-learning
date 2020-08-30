from matplotlib import pyplot
import numpy as np
from scipy import optimize
import utils

def costFunc(theta, X,y, lamda):
    m = y.size

    h = np.dot(X,theta)         #12*2 x 2*1

    J = (np.sum(np.square(h-y)))/(2*m) + (np.sum(np.square(theta[1:])))*(lamda/(2*m))

    grad = (1/m)*(h-y).dot(X)
    grad[1:] = grad[1:] + (lamda / m) * theta[1:]

    return J, grad


def trainLinearReg(X,y,lamda):
    initial_theta = np.zeros([X.shape[1],1])
    costFunction = lambda t: costFunc(t,X,y,lamda)
    options = {'maxiter':200}

    res = optimize.minimize(costFunction,initial_theta,jac = True, method = 'TNC', options=options)

    return res.x

def learningCurve(X,y,Xval,yval,lamda):

    m = y.size
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1,m+1):
        theta_t = trainLinearReg(X[:i,:], y[:i], lamda)
        error_train[i-1],_ = costFunc(theta_t, X[:i,:], y[:i], 0)
        error_val[i-1],_ = costFunc(theta_t, Xval, yval, 0)

    return error_train, error_val

def polyFeatures(X,p):
    m = X.size
    X_poly = np.zeros((m,p))
    for i in range(m):
        for j in range(p):
            X_poly[i,j] = X[i]**(j+1)

    return X_poly

def featureNormalize(X):
    mu= np.mean(X, axis = 0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis =0, ddof = 1)

    X_norm = X_norm / sigma

    return X_norm, mu, sigma

def polyFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma

    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)
    pyplot.plot(x, np.dot(X_poly, theta), '--', lw=2)

def validationCurve(X,y,Xval,yval):

    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))


    for i in range(len(lambda_vec)):
        lambda_try = lambda_vec[i]
        theta_t = trainLinearReg(X, y, lamda = lambda_try)
        error_train[i], _ = costFunc(theta_t, X, y, lamda = 0)
        error_val[i], _ = costFunc(theta_t, Xval, yval, lamda = 0)

    return lambda_vec, error_train, error_val
