# -*- coding: utf-8 -*-

import numpy as np
from pyemd import emd
import matplotlib.pyplot as plt
import multiprocessing as mp

# Earth Mover Distance
class emdFunc(object):
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, pair):
        return emd(pair[0], pair[1], self.dMat)

# metric-dependent width: E(d(X_1,X_2))
class varFunc(object):
    def __init__(self, dMat):
        self.dMat = dMat
    def __call__(self, p):
        pp = np.kron(p,p).reshape((len(p),len(p)))
        return np.sum(pp*self.dMat)
    
# information entropy
def ent(p):
    return -np.sum(p*np.log(p+1e-10))



"""
foldPrefix = "./data/IntSpin/"
#foldPrefix = "./data/ChaoSpin/"

p1t = np.loadtxt(foldPrefix+"p_0.98.dat")
p2t = np.loadtxt(foldPrefix+"p_0.99.dat")
p1cen = np.zeros(p1t.shape)
ind = 0
for j in np.argmax(p1t,axis=1):
    p1cen[ind,j] = 1.
    ind += 1

dMat = np.loadtxt(foldPrefix+"dMat.dat")

emdF = emdFunc(dMat)
varF = varFunc(dMat)
cores = mp.cpu_count()
pool = mp.Pool(processes = cores)

dist = list(pool.imap(emdF, zip(p1t,p2t)))
#dist = np.subtract(list(pool.imap(varF, p1t)),list(pool.imap(varF, p2t)))
#dist = np.subtract(list(pool.imap(ent, p1t)),list(pool.imap(ent, p2t)))
tlis = 0.1*np.arange(101)

pool.close()
pool.join()

plt.plot(tlis, dist)
"""
