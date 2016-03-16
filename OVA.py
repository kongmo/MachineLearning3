# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:34:05 2016

@author: aa
"""
import numpy as np
import scipy.optimize as sop
import CF


def oneVsAll(X,y,num_labels,Lambda):
    m=X.shape[0]
    n=X.shape[1]
    
    all_theta=np.zeros((num_labels,n+1))
    
    X=np.hstack((np.ones((m,1)),X))
    initial_theta=np.zeros(n+1)
    opt={'disp':True,'maxiter':50}

    for i in range(num_labels):
        args=(X, y==i+1 ,Lambda)
       # tmp=CF.lrcostFunction(initial_theta,X,y==i,Lambda,30)
        tmp=sop.minimize(CF.lrcostFunction,initial_theta,args=args,method='BFGS',options=opt)
        all_theta[i,:]= tmp.x
    return all_theta
