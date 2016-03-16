import numpy as np
import SM


def predictOneVsAll(all_theta,X):
    m=X.shape[0]
    X=np.hstack((np.ones((m,1)),X))
    all_theta=all_theta.transpose()

    tmp=SM.sigmoid(np.dot(X,all_theta))
    p=tmp.argmax(axis=1)
    print p
    p=p+1
    return p
        
