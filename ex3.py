import numpy as np
import scipy.io as sio
import DD
import OVA
import POVA

input_layer_size=400
num_labels=10

print 'Loading and Visualizing Data...'
data=sio.loadmat('ex3data1')
X=data['X']
y=data['y']
m=X.shape[0]

rand_indices=range(m)
np.random.shuffle(rand_indices)
sel=X[rand_indices[0:100],:]
#DD.displayData(sel)

# Part Two
print 'Training One-vs-All Logistic Regression...'
Lambda=10
all_theta=OVA.oneVsAll(X,y,num_labels,Lambda)

#Part Three
pred=POVA.predictOneVsAll(all_theta,X)
pred.shape=(pred.shape[0],1)
print 'Training Set Accuracry: %5.3f%%' % (np.mean(pred==y)*100)
