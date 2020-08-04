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

defec = 0.01
defecs = [0.01, 0.5]
defecSite = 1

j1 = 1.0
j2 = 0.5
jj1 = 1.
jj2 = 0.5

#alpha = np.float(sys.argv[1])
alpha = 0



seed = np.concatenate((np.ones(M),np.zeros(N-M)))
bsSet = np.array(list(multiset_permutations(seed)))
dim = len(bsSet)

plt.figure(figsize = (10,6))

for defec in defecs:
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
    
    eigs = sorted(eigs)
    
    
    esp = sc.eSpacingGet(eigs)
    xlab, ylab, wiglab, poilab = esp.T
    
    tmpx = xlab.copy()
    
    xlab = xlab[tmpx <= 7.]
    ylab = ylab[tmpx <= 7.]
    
    
    plt.plot(xlab, ylab, label = "Defect amplitude="+str(defec))
plt.plot(xlab, wiglab, label = "Wigner-like Spacing")
plt.plot(xlab, poilab, label = "Poisson-like Spacing")
plt.legend()
#plt.savefig(fold+prefix+"/Estat.png")