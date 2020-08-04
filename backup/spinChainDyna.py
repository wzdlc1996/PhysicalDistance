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
alpha = 0.05

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
    
sc.eSpacingGet(sorted(eigs))
    
# Physical distant matrix
dmat = np.zeros((dim,dim))
for i in range(dim):
    for j in range(i+1,dim):
        dmat[i,j] = sc.phyMeas((bsSet[i],bsSet[j]))
        dmat[j,i] = dmat[i,j]

cent = [5.3,5.7]
wid = [0.1,0.1]
tlis = np.arange(0.,10.,0.2)
inis = sc.gausIni(cent,wid,bsSet,eigv)
evos = sc.stateEvo(inis, eigs, tlis)

plis = [np.transpose(np.abs(np.dot(eigv, x))**2) for x in evos]
dlis = [sc.emd(x[0],x[1],dmat) for x in plis]
sc.plt.figure()
sc.plt.plot(dlis)


    

