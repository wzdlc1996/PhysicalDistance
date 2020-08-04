# -*- coding: utf-8 -*-

import numpy as np
import scSys as sc
from sympy.utilities.iterables import multiset_permutations
import os
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt

N = 6
M = 3

defec = 0.0
defecSite = 0
j1 = 1.0
j2 = 0.0
jj1 = 1.
jj2 = 0.5
#alpha = np.float(sys.argv[1])
alpha = 0.

seed = np.concatenate((np.ones(M),np.zeros(N-M)))
bsSet = np.array(list(multiset_permutations(seed)))
dim = len(bsSet)

# On-site energy and defect
hamil = np.zeros((dim,dim))
for i in range(dim):
    if bsSet[i,defecSite] == 1:
        hamil[i,i] += defec / 2
    else:
        hamil[i,i] -= defec / 2
for i in range(dim):
    for j in range(N):
        if bsSet[i,j] == bsSet[i,(j+1) % N]:
        #if bsSet[i,j] == bsSet[i,j+1]:
            hamil[i,i] += j2 / 4
        else:
            hamil[i,i] -= j2 / 4
# Nearest neighbor interaction
for i in range(dim):
    for j in range(i+1,dim):
        diffSite = sorted(np.nonzero(bsSet[i]-bsSet[j])[0])
        if len(diffSite) == 2 and (diffSite[1] - diffSite[0] in [1,N-1]):
            hamil[i,j] = j1 / 2
            hamil[j,i] = j1 / 2
# Next-nearest-neighbor interaction and On-site correction
if alpha != 0:
    for i in range(dim):
        for j in range(N):
            if bsSet[i,j] == bsSet[i,(j+2) % N]:
                hamil[i,i] += alpha * jj2 / 4
            else:
                hamil[i,i] -= alpha * jj2 / 4
    for i in range(dim):
        for j in range(i+1,dim):
            diffSite = sorted(np.nonzero(bsSet[i]-bsSet[j])[0])
            if len(diffSite) == 2 and (diffSite[1] - diffSite[0] in [2,N-2]):
                hamil[i,j] = alpha * jj1 / 2
                hamil[j,i] = alpha * jj1 / 2


eigs, eigv = np.linalg.eigh(hamil)

ea = np.sum(np.cos([np.pi*(M+1-2*(l+1))/N for l in range(M)]))

eb = np.cos([2*np.pi*(n/N) for n in range(N)])
ep = [np.sum(eb[np.nonzero(x)[0]]) for x in bsSet]


#print(hamil)