# -*- coding: utf-8 -*-

import numpy as np
import qkr
import scipy.special as sp
import os
import sys
import multiprocessing as mp

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

prefix = "./data/"
foldPref = prefix+"k_"+str(k)+"m_"+str(m)
try:
    os.mkdir(foldPref)
    ax = np.arange(0.,m)*dq
    Qs = np.tile(np.arange(0,m)*dq,(m,1)).T
    Ps = np.tile(np.arange(0,m)*dp,(m,1))
    xpGrid = np.array([[ax[i],ax[j]] for i in range(m) for j in range(m)])
    
    hamil = np.sum([blc(j) for j in np.arange(-int(3.*k/np.pi),1 + int(3.*k/np.pi))], axis=0)
    
    eigs = np.linalg.eig(hamil)
    
    eigVal = eigs[0]
    eigVec = eigs[1].T
    
    disMat = np.zeros((m**2,m**2))
    for i in range(m**2):
        for j in range(m**2):
            disMat[i,j] = qkr.dist((xpGrid[i], xpGrid[j]))
    
    DMat = np.zeros((m**2, m**2))
    for i in range(m**2):
        for j in range(m**2):
            DMat[i,j] = qkr.geoDist((phi2Ph(eigVec[i]),phi2Ph(eigVec[j])))
    
    np.save(foldPref+"/hamil.npy",hamil)
    np.savetxt(foldPref+"/DMat.dat",DMat)
    np.save(foldPref+"/eigv.npy", eigVec)
    np.savetxt(foldPref+"/xpGrid.dat", xpGrid)

except FileExistsError:
    DMat = np.loadtxt(foldPref+"/DMat.dat")
    eigVec = np.load(foldPref+"/eigv.npy")
    xpGrid = np.savetxt(foldPref+"/xpGrid.dat", xpGrid)
    
def phi2ePh(phi):
    res = (np.conjugate(eigVec).dot(np.transpose([phi]))).flatten()
    return np.abs(res)**2 / np.sum(np.abs(res)**2)

def chaoIndic(pair):
    return qkr.emd(phi2ePh(gaus(pair[0],pair[1])), np.ones(m**2)/(m**2), DMat)

def effDim(pair):
    ph = phi2ePh(gaus(pair[0], pair[1]))
    return 1./ np.sum(ph**2)

def eDetail(pair):
    return phi2ePh(gaus(pair[0],pair[1]))

siz = 3*m
pairs = [[x*2.*np.pi/siz,y*2.*np.pi/siz] for x in range(siz) for y in range(siz)]

pool = mp.Pool(mp.cpu_count())
res = np.array(pool.map(chaoIndic, pairs))
resref = np.array(pool.map(effDim, pairs))
eDet = np.array(pool.map(eDetail, pairs))

pool.close()
pool.join()

np.savetxt(foldPref+"/inds.dat", res)
np.savetxt(foldPref+"/refInds.dat", resref)
np.savetxt(foldPref+"/pos.dat",pairs)
np.savetxt(foldPref+"/eDist.dat", eDet)
