import numpy as np
import SM

def predict(theta1,theta2,X):
    m=X.shape[0]
    num_labels=theta2.shape[0]
    p=np.zeros(m)

    a1=np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1,theta1.transpose())
    a2 = SM.sigmoid(z2)
    n = a2.shape[0]
    a2 = np.hstack((np.ones((n,1)),a2))
    z3=np.dot(a2,theta2.transpose())
    a3=SM.sigmoid(z3)

    p=a3.argmax(axis=1)
    p=p+1
    return p
    
