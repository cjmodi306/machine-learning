import numpy as np
#scientific and vector computation for python,
#basically matrix use

from matplotlib import pyplot as plt
#plotting data points

data = np.loadtxt('ex1data1.txt', delimiter=',')
X,y = data[:,0], data[:,1]

m = X.size

def plotData(x,y):
    plt.figure()
    plt.plot(x,y,'ro',ms=10,mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of city in  10,000s')

#plt.show(plotData(X,y))

X = np.stack([np.ones(m),X], axis=1)


def gradientDescent(x,y,t,alpha,num_iters):
    m = y.size
    J_history = []

    for i in range(num_iters):
        h = np.matmul(x, t)
        t = t-((alpha/m)*(np.matmul(np.transpose(x),h-y)))
        J_history.append(costCompute(x,y,t))
    print(t)
    global theta
    theta = t
    print(J_history)
    return theta, J_history

def costCompute(X,y,theta):
    m = y.size
    J = 0

    h = np.matmul(X,theta)
    error = np.square(h-y)
    esum = np.sum(error)

    J = (1/(2*m))*(esum)
    return J


theta = np.zeros(2)
print(costCompute(X,y,theta))

num_iters= 1500
alpha = 0.01

theta,J = gradientDescent(X,y,theta,alpha,num_iters)
print((np.matmul(([1,7]),theta))*10000)


