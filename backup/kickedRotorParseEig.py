# -*- coding: utf-8 -*-
import numpy as np
from qkr import *
import multiprocessing as mp
import sys
import datetime as dt

tol = 1e-6
K = np.float(sys.argv[1])
m = np.int(sys.argv[2])
print("Begin project of K="+str(K)+"_m="+str(m))
st = dt.datetime.now()
foldPrefix = "./data/KickedRotorEig/"+"K="+str(K)+"_m="+str(m)

dMat = np.loadtxt(foldPrefix+"/dMat.dat")
xpGrid = np.loadtxt(foldPrefix+"/xpGrid.dat")
grdLis = np.int_(np.loadtxt(foldPrefix+"/grdLis.dat"))
eigphs = np.loadtxt(foldPrefix+"/eigphs.dat")