import numpy as np
import ex6_utils
from scipy.io import loadmat
import winsound


#----------------------------------EMAIL SPAM-----------------------------

with open('emailSample1.txt') as fid:file_contents = fid.read()

word_indices  = ex6_utils.processEmail(file_contents)

#Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)

with open('emailSample1.txt') as fid:file_contents = fid.read()

word_indices  = ex6_utils.processEmail(file_contents)
features      = ex6_utils.emailFeatures(word_indices)

# Print Stats
print('\nLength of feature vector: %d' % len(features))
print(features)
print('Number of non-zero entries: %d' % sum(features > 0))

#----------------------------------------SVM ACCURACY--------------------------------------------------
C = 100

data = loadmat('spamTrain.mat')
X, y = data['X'], data['y'][:, 0]
model = ex6_utils.svmTrain(X, y, C, ex6_utils.linearKernel)

p = ex6_utils.svmPredict(model, X)
print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))

data = loadmat('spamTest.mat')
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
print('Evaluating the trained Linear SVM on a test set ...')
p = ex6_utils.svmPredict(model, Xtest)

print('Test Accuracy: %.2f' % (np.mean(p == ytest) * 100))

#--------------------------------------TRYING YOUR OWN EMAIL--------------------------------------------------------

filename = ('spamSample1.txt')

with open(filename) as fid:
    file_contents = fid.read()

word_indices = ex6_utils.processEmail(file_contents, verbose=False)
x = ex6_utils.emailFeatures(word_indices)
p = ex6_utils.svmPredict(model, x)

print('\nProcessed %s\nSpam Classification: %s' % (filename, 'spam' if p==1 else 'not spam'))

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)