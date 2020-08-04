# -*- coding: utf-8 -*-
import numpy as np
import pyemd
import multiprocessing as mp
import sys
import datetime as dt

tol = 1e-6

# Sparse emd
def emd(p1,p2,dmat):
    msk = (p1+p2) >tol
    usdp1 = p1[msk]
    usdp2 = p2[msk]
    usdd = np.transpose(dmat[msk])[msk]
    return pyemd.emd(usdp1,usdp2,usdd)

# Classical Distance
def dist(pair):
    diff = np.abs(pair[0]-pair[1])
    return np.sum(np.min([diff,2.*np.pi-diff],axis = 0))

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

def crossEnt(pair):
    return np.sum(0.5*(pair[0]*np.log((pair[0]+1e-6)/(pair[1]+1e-6))) + 0.5*(pair[1]*np.log((pair[1]+1e-6)/(pair[0]+1e-6))))


K = np.float(sys.argv[1])
m = np.int(sys.argv[2])
sampLen = np.int(sys.argv[3])
print("Begin project of K="+str(K)+"_m="+str(m))
st = dt.datetime.now()
foldPrefix = "./data/KickedRotor/"+"K="+str(K)+"_m="+str(m)

dMat = np.loadtxt(foldPrefix+"/dMat.dat")
xpGrid = np.loadtxt(foldPrefix+"/xpGrid.dat")
grdLis = np.int_(np.loadtxt(foldPrefix+"/grdLis.dat"))
tLis = np.loadtxt(foldPrefix+"/tLis.dat")

qGross = np.loadtxt(foldPrefix+"/grossData_q.txt")
fGross = np.loadtxt(foldPrefix+"/grossData_cF.txt")
gGross = np.loadtxt(foldPrefix+"/grossData_cG.txt")

def neighbor(lab):
    diff = np.vstack((np.tile([-1,0,1],3),np.repeat([-1,0,1],3))).T
    res = np.unique(np.clip(lab + diff,0,m-1), axis=0)
    return [x for x in res if (x!=lab).any()]

def dataRead(lab):
    if not lab in grdLis:
        raise ValueError("lab is not in grdLis")
    """
    # Old data-reading procedure
    qlis = np.loadtxt(foldPrefix+"/"+str(lab[0])+"_"+str(lab[1])+"_q.txt")
    cpos = np.int_(np.loadtxt(foldPrefix+"/"+str(lab[0])+"_"+str(lab[1])+"_cG.txt"))-1
    clisG = np.zeros(qlis.shape)
    for i in range(len(cpos)):
        clisG[i,cpos[i]-1] = 1.
    clisF = np.loadtxt(foldPrefix+"/"+str(lab[0])+"_"+str(lab[1])+"_q.txt")
    """
    pos = np.argwhere(np.all(grdLis == lab, axis=1))[0,0]
    qlis = qGross[pos*len(tLis):(pos+1)*len(tLis)]
    cpos = np.int_(gGross[pos])-1
    clisG = np.zeros(qlis.shape)
    for i in range(len(cpos)):
        clisG[i,cpos[i]-1] = 1.
    clisF = np.split(fGross[pos],len(tLis))
    return qlis,clisG,cpos,clisF

samp = []

for ind in range(sampLen):
    pair = np.random.choice(len(grdLis),2)
    samp.append([grdLis[pair[0]], grdLis[pair[1]]])
    
def disOfT(grdPair):
    qlis1, clisG1, cpos1, clisF1 = dataRead(grdPair[0])
    qlis2, clisG2, cpos2, clisF2 = dataRead(grdPair[1])
    qdis = np.zeros(len(tLis))
    klDis = np.zeros(len(tLis))
    gdis = np.zeros(len(tLis))
    fdis = np.zeros(len(tLis))
    for j in range(len(tLis)):
        qdis[j] = emd(qlis1[j],qlis2[j],dMat)
        klDis[j] = crossEnt((qlis1[j], qlis2[j]))
        gdis[j] = dMat[cpos1[j]][cpos2[j]]
        fdis[j] = dist((clisF1[j],clisF2[j]))
    return np.concatenate((qdis, gdis, fdis, klDis))

cores = 8
pool = mp.Pool(processes = cores)

tabs = np.array(list(pool.imap(disOfT, samp)))

pool.close()
pool.join()

ed = dt.datetime.now()

np.savetxt(foldPrefix + "/GrossStatTabs.txt",tabs)

print("Time used: "+str(ed-st)+" data saved.")