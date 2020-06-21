import numpy as np
import matplotlib.pyplot as pyplot
from scipy.io import loadmat
from scipy import optimize
import cv2

data = loadmat('ex4data1.mat')
X,y = data['X'], data['y'].ravel()

m = X.shape[0]

input_layer = 400
hidden_layer = 25
num_classifiers = 10
lamda = 1

y[y==10] = 0

#_________________sigmoid and sigmoid gradient_______________

def sigmoid(z):
    zet = np.exp(-z)
    g = 1/(1+zet)
    return g

def sigmoidGradient(x):

    g_dash = np.multiply(sigmoid(x), 1-sigmoid(x))
    return g_dash

#_________________Weights initialisation_________________

def initWeights(conn_in, conn_out):

    w = np.zeros((conn_out,1+conn_in))
    epsilon = 0.12
    w = np.random.rand(conn_out, 1 + conn_in) * 2 * epsilon - epsilon
    return w

theta1 = initWeights(input_layer, hidden_layer)     #25*400
theta2 = initWeights(hidden_layer, num_classifiers)     #26*10


theta = np.concatenate([np.reshape(theta1,(10025,1)), np.reshape(theta2,(260,1))])

#__________________Cost Calculation__________________

def costFunc(theta, input_layer ,hidden_layer,num_classifiers,X, y, lamda):

    theta1 = theta[:(input_layer+1)*hidden_layer]
    theta2 = theta[(input_layer+1)*hidden_layer:]

    theta1 = np.reshape(theta1,(hidden_layer,(input_layer+1)))
    theta2 = np.reshape(theta2,(num_classifiers,(hidden_layer+1)))

    m = X.shape[0]

    X = np.concatenate([np.ones((m,1)), X], axis=1)  #5000*401
    z2 = np.matmul(X,np.transpose(theta1))    #5000*25
    a2 = sigmoid(z2)    #5000*25

    a2 = np.concatenate([np.ones((m,1)), a2], axis=1)
    z3 = np.matmul(a2,np.transpose(theta2))       #5000*10
    a3 = sigmoid(z3)        #5000*10

    y_new = np.zeros((m, num_classifiers))

    theta1_grad = np.zeros((theta1.shape))
    theta2_grad = np.zeros((theta2.shape))#

    #Converting y from 5000*1 into 5000*10 matrix

    for i in range(m):
        y_new[i, y[i]] = 1


    J = ((1 / m) * (np.sum(np.sum(np.multiply(-y_new,np.log(a3))
                                  - np.multiply((1 - y_new),np.log(1 - a3)))))
         + (lamda / (2 * m)) * np.sum(np.sum((np.multiply(theta1[:, 1:], theta1[:, 1:]))) +
                                      np.sum(np.sum(np.multiply(theta2[:, 1:], theta2[:, 1:])))))

    #_________________________Back Propagation__________________

    for i in range(m):

        a1 = X[i,:]
        a1 = a1.reshape(1,input_layer+1)

        z2 = np.matmul(a1, np.transpose(theta1))        # 1*25
        a2 = sigmoid(z2)        # 1*25
        a2 = np.concatenate([np.ones((1,1)), a2], axis=1)       # 1*26

        z3 = np.matmul(a2, np.transpose(theta2))        # 1*10
        a3 = sigmoid(z3)    # 1*10

        delta3 = a3-y_new[i]   # 1*10

        z2 = np.concatenate([np.ones((1,1)), z2], axis=1)
        delta2 = np.matmul(delta3, np.multiply(theta2, sigmoidGradient(z2)))
        delta2 = delta2[:,1:]

        theta2_grad = theta2_grad + np.matmul(np.transpose(delta3),a2)
        theta1_grad = theta1_grad + np.matmul(np.transpose(delta2),a1)

    theta1_grad[:,0] = theta1_grad[:,0]/m
    theta1_grad[:,1:] = (theta1_grad[:,1:]/m) + (lamda/m)* theta1[:,1:]

    theta2_grad[:,0] = theta2_grad[:,0]/m
    theta2_grad[:,1:] = (theta2_grad[:,1:]/m) + (lamda/m)* theta2[:,1:]

    grad = np.concatenate([np.reshape(theta1_grad,(hidden_layer*(input_layer+1),1)),
                           np.reshape(theta2_grad,((hidden_layer+1)*num_classifiers,1))])
    print(J)
    return J, grad

J, grad = costFunc(theta,input_layer,hidden_layer,num_classifiers, X,y,lamda)

#_________________Iterations and Theta optimization_________________

options = {'maxiter': 100}

costFunction = lambda p: costFunc(p, input_layer, hidden_layer,
                                  num_classifiers,X,y,lamda)

result = optimize.minimize(fun=costFunction,
                           x0=theta,
                           jac = True,
                           method='TNC',options=options)

cost = result.fun
theta = result.x

theta1 = theta[:hidden_layer*(input_layer+1)]
theta2 = theta[hidden_layer*(input_layer+1):]

theta1 = np.reshape(theta1, (hidden_layer, input_layer+1))
theta2 = np.reshape(theta2, (num_classifiers, hidden_layer+1))

def predict(Theta1, Theta2,X):

    m = X.shape[0]
    X = np.concatenate([np.ones((m,1)), X], axis = 1) #X: 5000*401
    z2 = np.matmul(X,np.transpose(Theta1))  #5000*25

    a2 = sigmoid(z2)    #5000*25
    a2 = np.concatenate([np.ones((m,1)), a2], axis = 1) #5000*26


    z3 = np.matmul(a2, np.transpose(Theta2)) #5000*10
    h = sigmoid(z3)

    p = np.argmax(h, axis=1)
    return p


pred = predict(theta1, theta2, X)

print(np.mean(pred == y) * 100)


###_______________________________________________________________________________TESTING AN IMAGE______________________________________________________________________________________________________________________###

image = cv2.imread('pixel5.png')
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
image = np.transpose(image)
image = np.array(image, dtype=float)
image = np.reshape(image,(1,400))

def displayData(X, example_width=None, figsize=(10, 10)):
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

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

    pyplot.show()


displayData(image)


impred=predict(theta1,theta2,image)
print('The number in image is: {}'.format(*impred))