# -*- coding: utf-8 -*-

import numpy as np
import qutip as quantum
from itertools import permutations
import scipy.sparse as sp
import matplotlib.pyplot as plt

nZero = 1e-6
nSite = 10
subSpaceLen = 5

# Coefficients of Hamiltonian
dSite = 4 # The site of the defect
dEn = 1.3 # The defect magnitude
paraZ = np.zeros(nSite)
paraZ[dSite] = dEn
paraJx = 2. * np.ones(nSite - 1)
paraJy = 2. * np.ones(nSite - 1)
paraJz = np.ones(nSite - 1)

s0 = quantum.qeye(2)
sx = quantum.sigmax()
sy = quantum.sigmay()
sz = quantum.sigmaz()

sxList = []
syList = []
szList = []

# construct the spin operators site by site
for n in range(nSite):
    tmpOpList = []
    for m in range(nSite):
        tmpOpList.append(s0)
    tmpOpList[n] = sx
    sxList.append(quantum.tensor(tmpOpList))
    tmpOpList[n] = sy
    syList.append(quantum.tensor(tmpOpList))
    tmpOpList[n] = sz
    szList.append(quantum.tensor(tmpOpList)) 

# construct the Hamiltonian
Hamil = 0

for n in range(nSite):
    Hamil += paraZ[n] * szList[n]
for n in range(nSite - 1):
    Hamil += paraJx[n] * sxList[n] * sxList[n+1]
    Hamil += paraJy[n] * syList[n] * syList[n+1]
    Hamil += paraJz[n] * szList[n] * szList[n+1]

# construct the Hamiltonian in Sub-space
seed = np.append(np.ones(subSpaceLen),np.zeros(nSite-subSpaceLen))
subSpaceBasisMask = np.unique(list(permutations(seed)), axis=0)
subSpaceDim = len(subSpaceBasisMask)

subSpaceBasis = []
for mask in subSpaceBasisMask:
    subSpaceBasis.append(quantum.tensor(list(map(lambda x: quantum.basis(2,int(x)), mask))))

projector = sp.hstack([x.data for x in subSpaceBasis])
    
subSpaceHamil = quantum.Qobj(projector.conj().T.dot(Hamil.data).dot(projector))

eigs = subSpaceHamil.eigenenergies()
nEigs = len(eigs)

"""
# Density of state
binSize = 1.
eMin = np.min(eigs)
eMax = np.max(eigs)
dELis = eMin + binSize * np.arange(int((eMax-eMin)/binSize))
rhoLis = np.zeros(len(dELis)-1)
for i in range(len(dELis)-1):
    rhoLis[i] = len([x for x in eigs if dELis[i] <= x <= dELis[i+1]])
rhoLis = np.array(rhoLis)/sum(rhoLis * binSize)
xlab = np.linspace(eMin,eMax,100)
ylab = rhoLis[np.clip(np.int_((xlab - eMin)/binSize), 0, 47)]
plt.plot(xlab, ylab)
"""

# Normalized Energy Spacing
grp = 4
per = 0.1 * nEigs
half = int(per/2)
spc = []
for j in range(int((nEigs - per)/grp)):
    avg = (eigs[half + grp*j] - eigs[half + grp*(j-1)])/grp
    for i in range(grp*(j-1), grp*j):
        spc.append((eigs[half + i] - eigs[half + i -1])/avg)
binSize = 0.1
spcM = np.max(spc)
dELis = binSize * (np.arange(int(spcM/binSize)) - 1.)
spcLis = np.zeros(len(dELis) - 1)
for i in range(len(dELis) - 1):
    spcLis[i] = len([x for x in spc if dELis[i] <= x <= dELis[i+1]])
spcLis = np.array(spcLis)/sum(spcLis * binSize)
xlab = np.linspace(0.,spcM,200)
ylab = spcLis[np.clip(np.int_((xlab)/binSize), 0, len(spcLis) - 1)]
wiglab = np.pi * xlab/2. * np.exp(-np.pi * xlab**2 /4.)
poilab = np.exp(- xlab)
fig, ax = plt.subplots(figsize = (10,6))
ax.plot(xlab, ylab, label = "Energy Spacing")
ax.plot(xlab, wiglab, label = "Wigner-like Spacing")
ax.plot(xlab, poilab, label = "Poisson-like Spacing")

quantum.qsave()
