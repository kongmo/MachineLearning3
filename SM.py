# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:35:54 2016

@author: aa
"""
import numpy as np

def sigmoid(Z):
    return 1.0/(1.0+np.exp(-Z))
