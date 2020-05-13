import numpy as np
from scipy import optimize
from scipy.io import loadmat
import matplotlib.pyplot as plt

input_layer = 400
num_labels = 10

data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']
y[y == 10] = 0

m = y.size

print(m)

rand_indices = np.random.choice(m, 100)
sel = X[rand_indices, :]

def displayData(X, example_width=None, figsize=(10,10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

displayData(sel)
#plt.show()

theta = np.array([-2,-1,1,2], dtype=float)

X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
print(X_t)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3


def sigmoid(zet):
    g = 1/(1+np.exp(-zet))
    return g


def costFunc(theta,X,y,lambda_t):
    m = y.size
    z = np.matmul(X, theta)
    h = sigmoid(z)

    if y.dtype == bool:
        y = y.astype(int)

    theta_reg = theta
    theta_reg[0] = 0

    y = y.ravel()

    J = ((1 / m) * (np.matmul(np.transpose(np.log(h)), ((-1)*y)) - np.matmul(np.transpose(np.log(1 - h)), (1 - y)))) + (lambda_t/(2*m))*np.matmul(theta_reg.T,theta_reg)

    grad = ((1 / m) * (np.matmul(np.transpose(X), h - y))) + ((lambda_t/m)*theta_reg)
    return J, grad

J, grad = costFunc(theta,X_t,y_t,lambda_t)
print(J,grad)

def OnevsAll(X,y,num_labels,lambda_t):
    m,n = X.shape
    all_theta = np.zeros((num_labels,n+1))
    X = np.concatenate([np.ones((m,1)), X], axis=1)

    options = {'maxiter': 50}

    for c in range(num_labels):
        initial_theta = np.zeros([(n + 1),1])
        result = optimize.minimize(fun=costFunc,
                               x0=initial_theta,
                               args=(X, (y==c), lambda_t),
                               jac = True,
                               method='CG',options=options)
        all_theta[c,:] = result.x

    return all_theta


lambda_t = 0.1
all_theta = OnevsAll(X,y,num_labels,lambda_t)
print(all_theta.shape)

def predictOnevsAll(all_theta,X):
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    p = np.matmul(X,all_theta.T)


    pred = np.argmax(p,axis=1)
    pred = np.reshape(pred,(m,1))
    return pred

pred = predictOnevsAll(all_theta,X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))