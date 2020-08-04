# -*- coding: utf-8 -*-

import numpy as np

a = np.loadtxt("./data/stat/k_0.9m_30/inds.dat")
b = np.loadtxt("./data/stat/k_0.9m_30/pos.dat")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
x, y, z = np.transpose(np.hstack((b, np.transpose([a]))))
N = int(len(z)**.5)
z = z.reshape(N, N)
plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
        cmap=cm.hot, norm=LogNorm())
plt.colorbar()
plt.show()