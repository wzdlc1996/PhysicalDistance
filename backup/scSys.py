# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyemd

tol = 1e-6

def dosGet(eigs):
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
    plt.plot(xlab, ylab)
    return np.transpose([xlab,ylab])

def eSpacingGet(eigs):
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
    binSize = spcM / 30.
    dELis = binSize * (np.arange(int(spcM/binSize)) - 1.)
    spcLis = np.zeros(len(dELis) - 1)
    for i in range(len(dELis) - 1):
        spcLis[i] = len([x for x in spc if dELis[i] <= x <= dELis[i+1]])
    spcLis = np.array(spcLis)/sum(spcLis * binSize)
    xlab = np.linspace(0.,spcM, 100)
    ylab = spcLis[np.clip(np.int_((xlab)/binSize), 0, len(spcLis) - 1)]
    wiglab = np.pi * xlab/2. * np.exp(-np.pi * xlab**2 /4.)
    poilab = np.exp(- xlab)
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(xlab, ylab, label = "Energy Spacing")
    ax.plot(xlab, wiglab, label = "Wigner-like Spacing")
    ax.plot(xlab, poilab, label = "Poisson-like Spacing")
    return np.transpose([xlab,ylab,wiglab,poilab])

def harmingMeas(pair):
    return len(np.nonzero(pair[0]-pair[1])[0])

def phyMeas(pair):
    a = np.nonzero(pair[0])[0]
    b = np.nonzero(pair[1])[0]
    return np.sum(np.abs(b-a))
    
# Sparse emd
def emd(p1,p2,dmat):
    msk = (p1+p2) >tol
    usdp1 = p1[msk]
    usdp2 = p2[msk]
    usdd = np.transpose(dmat[msk])[msk]
    return pyemd.emd(usdp1,usdp2,usdd)

# Earth Mover Distance
class emdFunc(object):
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, pair):
        return emd(pair[0], pair[1], self.dMat)
    
class emdForEig(object):
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
    siteNum = len(bsSet[0])
    xlis = np.array([np.arange(0., siteNum, 1)])
    gauslis = np.transpose([np.arange(0.,siteNum,siteNum/num)])
    raw = np.exp(-(xlis - gauslis)**2/(2*width**2))
    stats = np.dot(bsSet, np.transpose(raw)) # stats is a matrix, each column is a Gaussian initial state, not normalized
    statsInEig = np.dot(eigv.T, stats)
    prob = np.array([np.abs(x)**2/ np.linalg.norm(x)**2 for x in np.transpose(statsInEig)])
    return prob, np.arange(0.,siteNum,siteNum/num)

# The effective occupation of the given probability distribution
def effDim(prob):
    return 1./np.sum(prob**2)

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


    
