# -*- coding: utf-8 -*-

import numpy as np
import scSys as sc
from sympy.utilities.iterables import multiset_permutations
import os
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt

N = 15
M = 5

defec = np.float(sys.argv[1])
defecSite = 1
j1 = 1.0
j2 = 0.5
jj1 = 1.
jj2 = 0.5
#alpha = np.float(sys.argv[1])
alpha = 0.

fold = "./data/"
prefix = "N="+str(N)+"_M="+str(M)+"_defec="+str(defec)
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
    print("Folder Exists, reading from the given one")
    bsSet = np.loadtxt(foldPref+"/basis.txt")
    eigs = np.loadtxt(foldPref+"/eigs.txt")
    eigv = np.loadtxt(foldPref+"/eigv.txt")
    dim = len(bsSet)

print("Start construct dmat")
dmat = np.zeros((dim,dim))
for i in range(dim):
    for j in range(i+1,dim):
        dmat[i,j] = sc.harmingMeas((bsSet[i],bsSet[j]))
        dmat[j,i] = dmat[i,j]
np.savetxt(foldPref+"/dmat.txt",dmat)

pairs = np.array([[i,j] for i in range(dim) for j in range(i+1,dim)])

seigs = sorted(eigs)

res = sc.dosGet(seigs)

esp = sc.eSpacingGet(seigs)
xlab, ylab, wiglab, poilab = esp.T

plt.figure(figsize = (10,6))
plt.plot(xlab, ylab, label = "Energy Spacing")
plt.plot(xlab, wiglab, label = "Wigner-like Spacing")
plt.plot(xlab, poilab, label = "Poisson-like Spacing")
plt.legend()
plt.savefig(fold+prefix+"/Estat.png")

np.savetxt(foldPref+"/DoSdata.txt", res)
np.savetxt(foldPref+"/EspacingData.txt", esp)

def chaoInd(prob):
    return sc.emd(prob, np.ones(dim)/dim, dmat)


print("Start construct initial states")
#gaus, pos = sc.spinGausGen(20, 0.3, bsSet, eigv)

gaus = sc.spinNatStatVecGen(M,bsSet,eigv)
pos = np.arange(0,len(gaus))

prbGaus = sc.longTimeAvg(gaus, eigs, eigv)

psiE = np.transpose(eigv.conj().T.dot(gaus.T))

pool = mp.Pool(processes = mp.cpu_count())
#ind = pool.map(chaoInd, prbGaus)
eff = pool.map(sc.effDimByVec, psiE)
pool.close()
pool.join()

np.savetxt(foldPref+"/pos.dat", pos)
#np.savetxt(foldPref+"/ind.dat", ind)
np.savetxt(foldPref+"/eff.dat", eff)

"""
plt.figure(figsize=(8,5))
plt.title("Chaos Indicator")
plt.plot(pos, (ind - np.average(ind))/(max(ind)-min(ind)), color="red", label="Chaos indicator")
plt.plot(pos, (eff-np.average(eff))/(max(eff)-min(eff)), color="blue", label="Effective dimension")
plt.legend(loc="upper right")
plt.xlabel("Position")
plt.ylabel("Value")
plt.savefig(fold+prefix+"/indcator.png")
plt.show()

print("Chaos Indicator Range: "+str((max(ind)-min(ind))/np.average(ind)))
print("Eff Dim Range: "+str((max(eff)-min(eff))/np.average(eff)))
"""
