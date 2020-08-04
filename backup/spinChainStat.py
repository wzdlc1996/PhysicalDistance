# -*- coding: utf-8 -*-

import numpy as np
import scSys as sc
from sympy.utilities.iterables import multiset_permutations
import os
import multiprocessing as mp
import sys

N = 12
M = 4

defec = 0.0
defecSite = 0
j1 = 1.0
j2 = 0.5
jj1 = 1.
jj2 = 0.5
#alpha = np.float(sys.argv[1])
alpha = 0.5

fold = "./data/spinChain/"
prefix = "N="+str(N)+"_M="+str(M)+"_alpha="+str(alpha)
foldPref = fold+prefix

try:
    os.mkdir(foldPref)
    """
    Generate the basis set
    """
    seed = np.concatenate((np.ones(M),np.zeros(N-M)))
    bsSet = np.array(list(multiset_permutations(seed)))
    dim = len(bsSet)
    
    """
    Generate the Hamiltonian
    """
    # On-site energy and defect
    hamil = np.zeros((dim,dim))
    for i in range(dim):
        if bsSet[i,defecSite] == 1:
            hamil[i,i] += defec / 2
        else:
            hamil[i,i] -= defec / 2
    for i in range(dim):
        for j in range(N-1):
            if bsSet[i,j] == bsSet[i,j+1]:
                hamil[i,i] += j2 / 4
            else:
                hamil[i,i] -= j2 / 4
    # Nearest neighbor interaction
    for i in range(dim):
        for j in range(i+1,dim):
            diffSite = np.nonzero(bsSet[i]-bsSet[j])[0]
            if len(diffSite) == 2 and diffSite[1] - diffSite[0] == 1:
                hamil[i,j] = j1 / 2
                hamil[j,i] = j1 / 2
    # Next-nearest-neighbor interaction and On-site correction
    if alpha != 0:
        for i in range(dim):
            for j in range(N-2):
                if bsSet[i,j] == bsSet[i,j+2]:
                    hamil[i,i] += alpha * jj2 / 4
                else:
                    hamil[i,i] -= alpha * jj2 / 4
        for i in range(dim):
            for j in range(i+1,dim):
                diffSite = np.nonzero(bsSet[i]-bsSet[j])[0]
                if len(diffSite) == 2 and diffSite[1] - diffSite[0] == 2:
                    hamil[i,j] = alpha * jj1 / 2
                    hamil[j,i] = alpha * jj1 / 2

    
    eigs, eigv = np.linalg.eigh(hamil)
    
    np.savetxt(foldPref+"/eigs.txt",eigs)
    np.savetxt(foldPref+"/eigv.txt",eigv)
    np.savetxt(foldPref+"/basis.txt",bsSet)
except FileExistsError:
    bsSet = np.loadtxt(foldPref+"/basis.txt")
    eigs = np.loadtxt(foldPref+"/eigs.txt")
    eigv = np.loadtxt(foldPref+"/eigv.txt")
    dim = len(bsSet)
    
dmat = np.zeros((dim,dim))
for i in range(dim):
    for j in range(i+1,dim):
        dmat[i,j] = sc.harmingMeas((bsSet[i],bsSet[j]))
        dmat[j,i] = dmat[i,j]
np.savetxt(foldPref+"/dmat.txt",dmat)

pairs = np.array([[i,j] for i in range(dim) for j in range(i+1,dim)])

selPairs = pairs[np.random.choice(len(pairs), 10)]

pool = mp.Pool(processes = mp.cpu_count())
emdLis = list(pool.imap(sc.emdForEig(dmat, eigv), selPairs))
pool.close()
pool.join()
klLis = []
for i,j in selPairs:
    p1 = np.abs(eigv[:,i])**2
    p2 = np.abs(eigv[:,j])**2
    klLis.append(sc.crossEnt((p1,p2)))
    
np.savetxt(foldPref+"/emd_kl.txt", np.transpose([emdLis,klLis]))


DMat = np.zeros((dim,dim))
for s in range(len(pairs)):
    i, j=pairs[s]
    p1 = np.abs(eigv[:,i])**2
    p2 = np.abs(eigv[:,j])**2
    DMat[i,j] = sc.crossEnt((p1,p2))
    DMat[j,i] = DMat[i,j]

np.savetxt(foldPref+"/DMat.txt", DMat)

eigs = sorted(eigs)

res = sc.dosGet(eigs)

esp = sc.eSpacingGet(eigs)

np.savetxt(foldPref+"/DoSdata.txt", res)
np.savetxt(foldPref+"/EspacingData.txt", esp)

def chaoInd(prob):
    return sc.emd(prob, np.ones(dim)/dim, DMat)

gaus, pos = sc.spinGausGen(20, 0.3, bsSet, eigv)
pool = mp.Pool(processes = mp.cpu_count())
ind = list(pool.imap(chaoInd, gaus))
eff = list(pool.imap(sc.effDim, gaus))
pool.close()
pool.join()

np.savetxt(foldPref+"/pos.dat", pos)
np.savetxt(foldPref+"/ind.dat", ind)
np.savetxt(foldPref+"/eff.dat", eff)
