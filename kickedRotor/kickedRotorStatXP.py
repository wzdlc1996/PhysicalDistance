# -*- coding: utf-8 -*-

import numpy as np
import qkr
import scipy.special as sp
import os
import sys
import multiprocessing as mp
import progressbar

"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
"""

#k = 3.0 # The kicking strength
#m = 30 # The resolution for the quantum phase-space

k = np.float(sys.argv[1])
m = np.int(sys.argv[2])

hbar = 2.*np.pi/ (m**2)
per = 3
dim = per*m**2 # The dimension of the Hilbert space, also the cutoff of momentum

dq = 2.*np.pi/m
dp = 2.*np.pi/m

def blc(l):
    bone = np.array(range(m**2))
    indMat = (bone + l * m**2) - np.transpose([bone])
    sign = (np.complex(0,-1))**indMat
    jvs = sp.jv(indMat, k/hbar)
    phs = np.exp(np.complex(0,-1) * bone**2 * hbar/2.)
    return np.multiply(np.multiply(sign,jvs), phs)

def phi2Ph(phi):
    res = []
    for p in range(m):
        usdphi = phi[p*m:m+p*m]
        res.append(np.abs(np.fft.fft(usdphi))**2)
    #return np.transpose(res)/m
    return (np.transpose(res)/m).flatten()
        
def gaus(q,p):
    qax = np.arange(m**2)*2.*np.pi/(m**2)
    psi = np.exp(-(qax - q)**2/(4.*(2.*np.pi/m)**2) + np.complex(0,1)*(qax-q)*p/hbar)
    phi = np.fft.fft(psi)
    return phi / (np.sqrt(np.sum(np.abs(phi)**2)))

prefix = "./data/stat/"
foldPref = prefix+"k_"+str(k)+"m_"+str(m)+"_xp"
try:
    os.mkdir(foldPref)
    ax = np.arange(0.,m)*dq
    Qs = np.tile(np.arange(0,m)*dq,(m,1)).T
    Ps = np.tile(np.arange(0,m)*dp,(m,1))
    xpGrid = np.array([[ax[i],ax[j]] for i in range(m) for j in range(m)])
    
    hamil = np.sum([blc(j) for j in np.arange(-int(3.*k/np.pi),1 + int(3.*k/np.pi))], axis=0)
    
    eigs = np.linalg.eig(hamil)
    
    eigVal = eigs[0]
    eigVec = eigs[1]
            
    disMat = np.zeros((m**2,m**2))
    for i in range(m**2):
        for j in range(m**2):
            disMat[i,j] = qkr.dist((xpGrid[i], xpGrid[j]))
            
    pool = mp.Pool(mp.cpu_count())
    eigProbs = np.array(pool.map(phi2Ph, eigVec))
    pool.close()
    
    np.save(foldPref+"/hamil.npy",hamil)
    np.savetxt(foldPref+"/disMat.dat",disMat)
    np.save(foldPref+"/eigs.npy", eigVal)
    np.save(foldPref+"/eigv.npy", eigVec)
    np.savetxt(foldPref+"/xpGrid.dat", xpGrid)
    np.save(foldPref+"/eigPrb.npy", eigProbs)
    

except FileExistsError:
    eigVal = np.load(foldPref+"/eigs.npy")
    eigVec = np.load(foldPref+"/eigv.npy")
    xpGrid = np.loadtxt(foldPref+"/xpGrid.dat")
    disMat = np.loadtxt(foldPref+"/disMat.dat")
    eigProbs = np.load(foldPref+"/eigPrb.npy")
    
def psi2ePh(inis, eigv):
    return np.dot(eigv.conj().T, inis.T)

def chaoIndic(prob):
    return qkr.emd(prob, np.ones(m**2)/(m**2), disMat)
    
def effDim(pair):
    res = (np.conjugate(eigVec.T).dot(np.transpose([gaus(pair[0], pair[1])]))).flatten()
    res = np.abs(res)**2 / np.sum(np.abs(res)**2)
    return 1./ np.sum(res**2)

siz = m
pairs = [[x*2.*np.pi/siz,y*2.*np.pi/siz] for x in range(siz) for y in range(siz)]

gausVecs = np.array([gaus(p[0], p[1]) for p in pairs])

gausTimeAvg = qkr.longTimeAvg(gausVecs, eigVal, eigVec)

coreNum = np.minimum(mp.cpu_count(), 6)
totLoopLen = np.int(len(pairs) / coreNum)
res = []
pool = mp.Pool(coreNum)
for ind in progressbar.progressbar(range(0, len(pairs), coreNum)):
    resblock = pool.map(chaoIndic, gausTimeAvg[ind : np.minimum(ind + coreNum, len(pairs))])
    res += resblock

resref = np.array(pool.map(effDim, pairs))

pool.close()
pool.join()

np.savetxt(foldPref+"/inds.dat", res)
np.savetxt(foldPref+"/refInds.dat", resref)
np.savetxt(foldPref+"/pos.dat",pairs)
