# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import pyemd

tol = 1e-6

# Sparse emd
def emd(p1,p2,dmat):
    msk = (p1+p2) >tol
    usdp1 = p1[msk]
    usdp2 = p2[msk]
    usdd = np.transpose(dmat[msk])[msk]
    return pyemd.emd(usdp1,usdp2,usdd)

class emdFunc(object):
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, pair):
        return emd(pair[0], pair[1], self.dMat)
    
# metric-dependent width: E(d(X_1,X_2))
class varFunc(object):
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, p):
        pp = np.kron(p,p).reshape((len(p),len(p)))
        return np.sum(pp*self.dMat)
    
# information entropy
def ent(p):
    return -np.sum(p*np.log(p+1e-10))

def crossEnt(pair):
    return np.sum(0.5*(pair[0]*np.log((pair[0]+1e-6)/(pair[1]+1e-6))) + 0.5*(pair[1]*np.log((pair[1]+1e-6)/(pair[0]+1e-6))))

# Kimetic distance
def geoDist(pair):
    return np.sum((pair[0]-pair[1])**2)

# Classical Distance
def dist(pair):
    diff = np.abs(pair[0]-pair[1])
    return np.sum(np.min([diff,2.*np.pi-diff],axis = 0))

# Some function of Floquet Operator construction

def sparsBesJ(n, x):
    if np.abs(n) > 10. * x:
        return 0.
    else:
        return sp.jv(n,x)
    
