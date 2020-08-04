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
    diff = np.abs(np.subtract(pair[0],pair[1]))
    return np.sqrt(np.sum(np.min([diff,2.*np.pi-diff],axis = 0)**2))

# Some function of Floquet Operator construction

def sparsBesJ(n, x):
    if np.abs(n) > 10. * x:
        return 0.
    else:
        return sp.jv(n,x)
    
def longTimeAvg(inis, eigs, eigv):
    """
    Return the long time average distribution with given initial states, eigen
    -values, and eigenvectors.
    """
    dim = len(eigs) # Dimension of the Hilbert space
    deltMat = np.zeros((dim, dim))
    for i in range(dim):
        deltMat[i,i] = 1
        for j in range(i+1, dim):
            if np.abs(eigs[i] * np.conj(eigs[j]) - 1) <= tol:
                deltMat[i,j] = 1
                deltMat[j,i] = 1
    psiE = np.conj(eigv.T).dot(inis.T)
    res = np.einsum("ij,kj,ji,ka,aj->ai",eigv,deltMat,eigv.conj().T,psiE,psiE.conj().T, optimize=True)
    z = np.max(np.abs(np.imag(res)))
    if z > tol:
        print("Warning: Result distribution has non-neglectable imaginary part, maximum: "+str(z)+ " , check code again!")
    return np.real(res)
    
