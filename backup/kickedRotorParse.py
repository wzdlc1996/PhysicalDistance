# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp
import sys
import datetime as dt
from qkr import *

tol = 1e-6

K = np.float(sys.argv[1])
m = np.int(sys.argv[2])
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

def meanDis(lab):
    neig = neighbor(lab)
    qlis,clisG,cpos,clisF = dataRead(lab)
    qdis = np.zeros(len(tLis))
    kldis = np.zeros(len(tLis))
    gdis = np.zeros(len(tLis))
    fdis = np.zeros(len(tLis))
    for x in neig:
        qlisRef,clisGRef,cposRef,clisFRef = dataRead(x)
        for j in range(len(tLis)):
            qdis[j] += emd(qlis[j],qlisRef[j],dMat)
            kldis[j] += crossEnt((qlis[j],qlisRef[j]))
            gdis[j] += dMat[cpos[j]][cposRef[j]]
            fdis[j] += dist((clisF[j],clisFRef[j]))
    qdis /= len(neig)
    gdis /= len(neig)
    fdis /= len(neig)
    return qdis, gdis, fdis, kldis

def qdisTimeAvg(lab):
    qdis, gdis, fdis = meanDis(lab)
    return np.average(qdis)

def fdisTimeAvg(lab):
    qdis, gdis, fdis = meanDis(lab)
    return np.average(fdis)

def grossDis(lab):
    return np.concatenate(meanDis(lab))

def qTimeAvgDis(lab):
    neig = neighbor(lab)
    qlis,clisG,cpos,clisF = dataRead(lab)
    qdis = 0.
    for x in neig:
        qlisRef,clisGRef,cposRef,clisFRef = dataRead(x)
        qdis += emd(np.average(qlis,axis=0),np.average(qlisRef,axis=0),dMat)
    qdis /= len(neig)
    return qdis

cores = 8
pool = mp.Pool(processes = cores)

tabs = list(pool.imap(grossDis, grdLis))
qAVtab = list(pool.imap(qTimeAvgDis, grdLis))

pool.close()
pool.join()

ed = dt.datetime.now()

np.savetxt(foldPrefix + "/GrossTabs.txt",np.hstack((xpGrid, np.array(tabs))))
np.savetxt(foldPrefix + "/qAVtab.txt",np.hstack((xpGrid, np.transpose([qAVtab]))))

print("Time used: "+str(ed-st)+" data saved.")