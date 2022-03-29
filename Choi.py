# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:01:41 2022

@author: mason
"""

import itertools
import numpy as np

def basis_vec(d):
    basis = []
    for i in range(d):
        vec = np.zeros((d,1))
        vec[i] = int(1)
        basis.append(vec)
    return basis

def depol(X,d,q):
    return (1-q)*X+q*np.trace(X)*np.identity(d)/d
def wer_hol(X,d,q):
    return (1-q)*np.transpose(X)+q*np.trace(X)*np.identity(d)/d
        
def choi(d,q,channel):
    
    basis = []
    for i in range(d):
        basis.append(i)
    a = list(itertools.product(basis,repeat=2))
    
    choi = np.zeros((d**2,d**2))
    
    for pair in a:
        i = basis_vec(d)[pair[0]]
        
        j = np.transpose(basis_vec(d)[pair[1]])
        
        tensor1 = np.dot(i,j)
        
        tensor2 = channel(tensor1,d,q)
        
        choi += np.kron(tensor1,tensor2)
    return choi


print(choi(2,1/2,depol))

