# -*- coding: utf-8 -*-

"""
Some core functions for spin-chain system.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyemd

tol = 1e-6

def dosGet(eigs):
    """
    Compute the Density of State(dos) from the given eigen-energy list:

        dos(E)dE = StateNumberInRange([E,E + dE])

    Input:
        eigs: np.array of size (Dim,) eigenvalue of Hamiltonian

    Output:
        res: np.array of size (100,2) containing the information of the dos funct
             -ion:

             res[:,0]: the different energy value of range [min(eigs), max(eigs)]
                       of length 100
             res[:,1]: the dos at energy res[:,0]
    """
    eMin = np.min(eigs)
    eMax = np.max(eigs)
    binSize = 10.* (eMax-eMin) / (len(eigs))
    dELis = eMin + binSize * np.arange(int((eMax-eMin)/binSize))
    rhoLis = np.zeros(len(dELis)-1)
    for i in range(len(dELis)-1):
        rhoLis[i] = len([x for x in eigs if dELis[i] <= x <= dELis[i+1]])
    rhoLis = np.array(rhoLis)/sum(rhoLis * binSize)
    xlab = np.linspace(eMin,eMax,100)
    ylab = rhoLis[np.clip(np.int_((xlab - eMin)/binSize), 0, len(rhoLis)-1)]
    #plt.plot(xlab, ylab)
    return np.transpose([xlab,ylab])

def eSpacingGet(eigs):
    """
    Compute the energy spacing value together with two standard spacing distrib
    -utions.

        eSpacing(Delta)dDelta = NumberOfSpacingInRange([Delta, Delta + dDelta])

    Input:
        eigs: np.array of size (Dim,) eigenvalue of Hamiltonian

    Output:
        res: np.array of size (100,4) containig the information of spacing dist
             -ributions.

             res[:,0]: the spacing value list
             res[:,1]: the density of spacing value at res[:,0]
             res[:,2]: the density of spacing value of Wigner-like distribution
             res[:,3]: the density of spacing value of Poisson-like distribution
    """
    grp = 4
    nEigs = len(eigs)
    per = 0.1 * nEigs
    half = int(per/2)
    spc = []
    for j in range(int((nEigs - per)/grp)):
        avg = (eigs[half + grp*j] - eigs[half + grp*(j-1)])/grp
        for i in range(grp*(j-1), grp*j):
            spc.append((eigs[half + i] - eigs[half + i -1])/avg)
    spcM = np.max(spc)
    binSize = 0.1
    dELis = binSize * (np.arange(int(spcM/binSize)) - 1.)
    spcLis = np.zeros(len(dELis) - 1)
    for i in range(len(dELis) - 1):
        spcLis[i] = len([x for x in spc if dELis[i] <= x <= dELis[i+1]])
    spcLis = np.array(spcLis)/sum(spcLis * binSize)
    xlab = np.linspace(0.,spcM, 100)
    ylab = spcLis[np.clip(np.int_((xlab)/binSize), 0, len(spcLis) - 1)]
    wiglab = np.pi * xlab/2. * np.exp(-np.pi * xlab**2 /4.)
    poilab = np.exp(- xlab)
    #plt.figure(figsize = (10,6))
    #plt.plot(xlab, ylab, label = "Energy Spacing")
    #plt.plot(xlab, wiglab, label = "Wigner-like Spacing")
    #plt.plot(xlab, poilab, label = "Poisson-like Spacing")
    return np.transpose([xlab,ylab,wiglab,poilab])

def harmingMeas(pair):
    """
    Compute the Harming Measurement D between two 0/1 valued lists who
    have the same length:

        D(x, y) is number of indices on which x and y's values are different

    Example:
        D( (1,1,0,0,1), (1,0,0,1,1) ) = 2
        D( (1,1,0,0,0), (0,0,0,1,1) ) = 4

    Input:
        pair: np.array of size (2, N), pair[0] and pair[1] are two 0/1 valued l
        -ist

    Output:
        D(pair[0], pair[1])
    """
    return len(np.nonzero(pair[0]-pair[1])[0])

def phyMeas(pair):
    """
    Compute the modified Harming distance between two 0/1 valued lists who have
    the same number of 1 and same length, which is better match the physics mea
    -ning than the original one:

        Dphy(x, y) is the sum of distances between each 1s in two lists

    Example:
        Dphy( (1,1,0,0,1), (1,0,0,1,1) ) = 2
        Dphy( (1,1,0,0,0), (0,0,0,1,1) ) = 6

    Input:
        pair: np.array of size (2, N), pair[0] and pair[1] are two 0/1 valued l
        -ist, pair[0] and pair[1] should have the same number of 1s

    Output:
        Dphy(pair[0], pair[1])
    """
    a = np.nonzero(pair[0])[0]
    b = np.nonzero(pair[1])[0]
    return np.sum(np.abs(b-a))

# Sparse emd
def emd(p1,p2,dmat):
    """
    Compute the sparsing Earth-Mover distance between two descreted distributio
    -ns: p1, p2 with the metric described by: dmat
    Our sparsing algorithm is:
        Use those points on which p1 + p2 value is greater than tol (1e-6 defau
        -lt) to construct two new distributions and compute their emd. The reas
        -on is that the emd will only depends on those points on which p1, p2 a
        -re not too small.

    Input:
        p1: 1d non-negative np.array of size (N,) of a flattened distribution
        p2: 1d non-negative np.array of size (N,) of a flattened distribution
        dmat: np.array of size (N,N), dmat[i,j] is the distance between the i-th
              and j-th points

    Output:
        The Earth-Mover distance between p1, p2 with metric dmat
    """
    msk = (p1+p2) >tol
    usdp1 = p1[msk]
    usdp2 = p2[msk]
    usdd = np.transpose(dmat[msk])[msk]
    return pyemd.emd(usdp1,usdp2,usdd)

# Earth Mover Distance
class emdFunc(object):
    """
    Used to compute the emd parallely
    """
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, pair):
        return emd(pair[0], pair[1], self.dMat)

class emdForEig(object):
    """
    Used to compute the emd parallely for energy-eigen states
    """
    def __init__(self, dMat, eigv):
        self.dmat = dMat
        self.eigv = eigv
    def __call__(self, pair):
        p1 = np.abs(self.eigv[:,pair[0]])**2
        p2 = np.abs(self.eigv[:,pair[1]])**2
        return emd(p1, p2, self.dmat)

# metric-dependent width: E(d(X_1,X_2))
class varFunc(object):
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, p):
        pp = np.kron(p,p).reshape((len(p),len(p)))
        return np.sum(pp*self.dMat)

# information entropy
def ent(p):
    return -np.sum(p*np.log(p+tol))

def crossEnt(pair):
    return np.sum(0.5*(pair[0]*np.log((pair[0]+tol)/(pair[1]+tol))) + 0.5*(pair[1]*np.log((pair[1]+tol)/(pair[0]+tol))))

def spinGausGen(num, width, bsSet, eigv):
    """
    Generate a set of probability distributions of states who are gaussian megn
    -itized along the z-direction, whose centers are uniformly distributed on
    the chain and width are given in parameters

        <b_i|psi> ~ b_i . Gauss

    Where Gauss is the descretized Gaussian distribution on the chain, b_i is
    the natural basis of 0/1 valued list
    """
    siteNum = len(bsSet[0])
    xlis = np.array([np.arange(0., siteNum, 1)])
    gauslis = np.transpose([np.arange(0.,siteNum,siteNum/num)])
    raw = np.exp(-(xlis - gauslis)**2/(2*width**2))
    rawT = raw.T / np.sum(raw.T, axis=0)
    stats = np.dot(bsSet, rawT) # stats is a matrix, each column is a Gaussian initial state, not normalized
    statsInEig = np.dot(eigv.T, stats)
    prob = np.array([np.abs(x)**2/ np.linalg.norm(x)**2 for x in np.transpose(statsInEig)])
    return prob, np.arange(0.,siteNum,siteNum/num)

def spinPerdGausGen(num, width, bsSet, eigv):
    siteNum = len(bsSet[0])
    xlis = np.array([np.arange(-siteNum, 2*siteNum, 1.)])
    gauslis = np.transpose([np.arange(0.,siteNum,siteNum/num)])
    rawFull = np.exp(-(xlis - gauslis)**2/(2*width**2))
    raw = rawFull[:,0:siteNum]+rawFull[:,siteNum:2*siteNum]+rawFull[:,2*siteNum:]
    rawT = raw.T / np.sum(raw.T, axis=0)
    stats = np.dot(bsSet, rawT) # stats is a matrix, each column is a Gaussian initial state, not normalized
    statsInEig = np.dot(eigv.T, stats)
    prob = np.array([np.abs(x)**2/ np.linalg.norm(x)**2 for x in np.transpose(statsInEig)])
    return prob, np.arange(0.,siteNum,siteNum/num)

def spinNatGen(upNum, bsSet, eigv):
    siteNum = len(bsSet[0])
    locs = np.array([[0]*x+[1]*upNum+[0]*(siteNum-upNum-x) for x in range(siteNum-upNum+1)])
    locState = np.array([np.float_(np.all(bsSet==z,axis=1)) for z in locs])
    return np.array([np.abs(x)**2/np.linalg.norm(x)**2 for x in np.dot(locState, eigv)])

def spinPerdNatGen(upNum, bsSet, eigv):
    siteNum = len(bsSet[0])
    locs = np.array([[0]*x+[1]*upNum+[0]*(siteNum-upNum-x) for x in range(siteNum-upNum+1)])
    for z in range(1,upNum):
        locs = np.concatenate((locs, np.array([[1]*z+[0]*(siteNum-upNum)+[1]*(upNum-z)])))
    locState = np.array([np.float_(np.all(bsSet==z,axis=1)) for z in locs])
    return np.array([np.abs(x)**2/np.linalg.norm(x)**2 for x in np.dot(locState, eigv)])

def spinNatStatVecGen(upNum, bsSet, eigv):
    siteNum = len(bsSet[0])
    locs = np.array([[0]*x+[1]*upNum+[0]*(siteNum-upNum-x) for x in range(siteNum-upNum+1)])
    locState = np.array([np.float_(np.all(bsSet==z,axis=1)) for z in locs])
    return locState

def spinPerdNatStatVecGen(upNum, bsSet, eigv):
    siteNum = len(bsSet[0])
    locs = np.array([[0]*x+[1]*upNum+[0]*(siteNum-upNum-x) for x in range(siteNum-upNum+1)])
    for z in range(1,upNum):
        locs = np.concatenate((locs, np.array([[1]*z+[0]*(siteNum-upNum)+[1]*(upNum-z)])))
    locState = np.array([np.float_(np.all(bsSet==z,axis=1)) for z in locs])
    return locState

# The effective occupation of the given probability distribution
def effDim(prob):
    return 1./np.sum(prob**2)

def effDimByVec(psi):
    prob = np.abs(psi)**2 / np.sum(np.abs(psi)**2)
    return effDim(prob)

# Return Gaussian initial state in energy-eigen basis, stored in each column
# cents is an array of centers of states and widths is an array of widths
def gausIni(cents, widths, bsSet, eigv):
    siteNum = len(bsSet[0])
    xlis = np.array([np.arange(0., siteNum, 1)])
    gauslis = np.transpose([cents])
    raw = np.exp(-(xlis - gauslis)**2/(2* np.transpose([widths]) **2))
    stats = np.dot(bsSet, np.transpose(raw)) # stats is a matrix, each column is a Gaussian initial state, not normalized
    statsInEig = np.dot(eigv.T, stats)
    st = np.array([x / np.linalg.norm(x) for x in np.transpose(statsInEig)])
    return st.T

# Evolution of initial states in energy basis
def stateEvo(inis, eigs, tlis):
    res = []
    for t in tlis:
        evo = np.diagflat(np.exp(-eigs * 1j * t))
        res.append(np.dot(evo, inis))
    return res

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
            if np.abs(eigs[i]-eigs[j]) <= tol:
                deltMat[i,j] = 1
                deltMat[j,i] = 1
    psiE = np.conj(eigv.T).dot(inis.T)
    res = np.einsum("ij,kj,ji,ka,aj->ai",eigv,deltMat,eigv.conj().T,psiE,psiE.conj().T, optimize=True)
    z = np.max(np.abs(np.imag(res)))
    if z > tol:
        print("Warning: Result distribution has non-neglectable imaginary part, maximum: "+str(z)+ " , check code again!")
    return np.real(res)
