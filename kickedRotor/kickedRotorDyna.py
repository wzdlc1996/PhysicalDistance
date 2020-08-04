# -*- coding: utf-8 -*-

import numpy as np
import qkr
import scipy.special as sp
import sys
import os
import multiprocessing as mp
import matplotlib.pyplot as plt


#k = 1.5 # The kicking strength
#m = 30 # The resolution for the quantum phase-space

k = np.float(sys.argv[1])
m = np.int(sys.argv[2])

hbar = 2.*np.pi/ (m**2)
per = 3. # The precision of cutoff.

dq = 2.*np.pi/m
dp = 2.*np.pi/m

ini = np.array([[4.7,3.],[4.7+dq,3.+dp]])

qlis = np.array([[ini[0,0], ini[1,0]]])
plis = np.array([[ini[0,1],ini[1,1]]])

trajLen = 50

"""
Classical Evolution
"""

for ki in range(trajLen):
    plis = np.concatenate((plis, [np.mod(plis[-1]+k * np.sin(qlis[-1]), 2.*np.pi)]))
    qlis = np.concatenate((qlis, [np.mod(qlis[-1]+plis[-1], 2.*np.pi)]))
st1 = np.transpose([qlis[:,0],plis[:,0]])
st2 = np.transpose([qlis[:,1],plis[:,1]])
clis = [qkr.dist((s1,s2)) for s1,s2 in zip(st1,st2)]

"""
Quantal Evolution
"""

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
    hamil = np.load(foldPref+"/hamil.npy")

ax = np.arange(0.,m)*dq
xpGrid = np.array([[ax[i],ax[j]] for i in range(m) for j in range(m)])
dmat = np.zeros((m**2,m**2))
for i in range(m**2):
    for j in range(i+1, m**2):
        dmat[i,j] = qkr.dist((xpGrid[i], xpGrid[j]))
        dmat[j,i] = dmat[i,j]

psis = np.array([np.array([gaus(ini[0,0],ini[0,1]),gaus(ini[1,0],ini[1,1])]).T])

for j in range(trajLen):
    psis = np.concatenate((psis, [np.dot(hamil, psis[-1])]))
psi1 = psis[:,:,0]
psi2 = psis[:,:,1]

p1t = np.array([phi2Ph(phi) for phi in psi1])
p2t = np.array([phi2Ph(phi) for phi in psi2])

def mpEMD(pair):
    return qkr.emd(pair[0],pair[1],dmat)

def Edis(pair):
    q1 = np.sum(pair[0]*xpGrid[:,0])
    p1 = np.sum(pair[0]*xpGrid[:,1])
    q2 = np.sum(pair[1]*xpGrid[:,0])
    p2 = np.sum(pair[1]*xpGrid[:,1])
    return qkr.dist(([q1,p1],[q2,p2]))

def mpFunc(pair):
    return mpEMD(pair), Edis(pair)

pairs = zip(p1t,p2t)

cores = mp.cpu_count()
pool = mp.Pool()
zlis = np.array(list(pool.imap(mpFunc, pairs)))
#elis = list(pool.imap(Edis, pairs))
pool.close()
pool.join()

qlis = zlis[:,0]
elis = zlis[:,1]

tlis = np.arange(0, trajLen+1)

np.savetxt(foldPref+"/qlis.dat", qlis)
np.savetxt(foldPref+"/elis.dat", elis)
np.savetxt(foldPref+"/tlis.dat", tlis)
np.savetxt(foldPref+"/clis.dat", clis)

plt.figure(figsize=(8,4))
plt.title("Classical/Quantum Corresponding")
plt.plot(tlis, clis, color="blue", label="classical")
plt.plot(tlis, qlis, color="red", label="quantalWass")
plt.plot(tlis, elis, color="green", label="quantalExp")
plt.xlabel("kicked Times")
plt.ylabel("distance")
plt.legend(loc = "upper right")
plt.savefig(foldPref+"/fig_k="+str(k)+"_m="+str(m)+".png")
plt.show()
