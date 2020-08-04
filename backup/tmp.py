#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:33:18 2019

@author: leonard
"""

import numpy as np
from pyemd import emd
import datetime as dt

a = np.loadtxt("/home/leonard/a.txt")
b = np.loadtxt("/home/leonard/b.txt")
d = np.loadtxt("/home/leonard/d.txt")
ae = np.loadtxt("/home/leonard/ae.txt")
be = np.loadtxt("/home/leonard/be.txt")
de = np.loadtxt("/home/leonard/de.txt")

st = dt.datetime.now()
p=emd(a,b,d)
ed = dt.datetime.now()
print("time used: "+str(ed-st))

st = dt.datetime.now()
q=emd(ae,be,de)
ed = dt.datetime.now()
print("time used: "+str(ed-st))

print((p-q)/q)