import scipy.io as sio
import DD
import numpy as np
import PD

input_layer_size=400
hidden_layer_size=25

num_labels=10

#Part One
print 'Loading and Visualizing Data...'
data=sio.loadmat('ex3data1')
X=data['X']
y=data['y']

m=X.shape[0]
sel=range(m)
np.random.shuffle(sel)
sel=sel[0:100]

#DD.displayData(X[sel,:])

#Part Two
print 'Loading Saved Neural Network Parameters...'

Theta=sio.loadmat('ex3weights')
Theta1=Theta['Theta1']
Theta2=Theta['Theta2']

pred=PD.predict(Theta1,Theta2,X)

pred.shape=(pred.shape[0],1)
print 'Training Set Accuracy: %5.3f%%' %(np.mean(pred==y)*100)
