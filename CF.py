import numpy as np
import SM


def lrcostFunction(theta,X,y,Lambda):
    m=y.shape[0]

    grad=np.zeros(theta.shape[0])
    tmp=SM.sigmoid(np.dot(X,theta))
    tmp.shape=(tmp.shape[0],1)
    aux1=-y*np.log(tmp+0.001)
    aux2=(1-y)*np.log(1-tmp+0.001)
    J=1.0/m*sum(aux1-aux2)+Lambda/(2*m)*sum(theta[1:]**2)

    grad[0]=1.0/m*X[:,0].dot(tmp-y)
    grad[1:]=1.0/m*sum((tmp-y)*X[:,1:])+Lambda/m*theta[1:]
    return J
                   
