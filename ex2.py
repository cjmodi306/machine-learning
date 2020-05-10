import numpy as np
#module for matrix operations

from scipy import optimize
#module to optimize the value of theta

#importing training data
data = np.loadtxt('ex2data1.txt', delimiter=',')
X, y = data[:, 0:2], data[:, 2]

m = y.size
n = X.shape[1]

#adding a roq of ones in X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

#initialise theta
initial_theta = np.zeros(n + 1)
test_theta = np.array([-24, 0.2, 0.2])

#sigmoid function for hypothesis in logical regression
def sigmoid(zet):
    a = np.exp(-zet)
    g = 1 / (1 + a)
    return g

#costfunction to minimize error
def costFunc(theta,X,y):
    m = y.size
    z = np.matmul(X, theta)
    h = sigmoid(z)
    J = (1 / m) * (np.matmul(np.transpose(np.log(h)), (-y)) - np.matmul(np.transpose(np.log(1 - h)), (1 - y)))
    grad = (1 / m) * (np.matmul(np.transpose(X), h - y))
    return J, grad


costFunc(initial_theta,X,y)
#print(J, grad)

#function to optimize value of theta
#this is alternative to 'fminunic' function in Octave
options = {'maxiter': 400}
result = optimize.minimize(fun=costFunc, x0=initial_theta, args=(X, y), jac = True, method='TNC',options=options)

cost = result.fun
theta = result.x

print(cost)
print(theta)

#predicting the accuracy
def predict(theta,X):
    z = np.matmul(X,theta)
    p = [sigmoid(z) >= 0.5]
    acc = np.mean(p ==y)
    print(acc*100)

#solving an example
predict(theta,X)
exam = np.array([1,45,85])
ans = np.matmul(exam,theta)

print(sigmoid(ans))
