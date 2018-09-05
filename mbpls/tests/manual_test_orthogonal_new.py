#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the package with orthogonal data

@author: Andreas Baum, andba@dtu.dk
@edited: Laurent Vermue, lauve@dtu.dk
"""

from matplotlib import pyplot as plt
from mbpls.data.get_data import data_path
from mbpls.mbpls import MBPLS
from mbpls.data.get_data import orthogonal_data
import numpy as np

plt.ioff()

x1, x2, x3, y = orthogonal_data(100)


def plotdata(spectra, yields):
    yields = yields + abs(yields.min())
    for spectrum, yield1 in zip(spectra, yields / yields.max()):
        plt.plot(spectrum.T, color=[0, 0, yield1], linewidth=6)


def preprocess(spectra, yields):
    from sklearn.preprocessing import StandardScaler
    Scaler1 = StandardScaler(with_mean=True, with_std=False)
    Scaler2 = StandardScaler(with_mean=True, with_std=True)
    return Scaler1.fit_transform(spectra), Scaler2.fit_transform(yields)


plt.close('all')
# x1, x2, y = loaddata()
# x1_process, y_process = preprocess(x1, y)
# x2_process, y_process = preprocess(x2, y)

# Plot preprocessed datablocks X1 and X2 colored according to y1, y2, y3
plt.subplot(311)
plotdata(x1, y[:, 0:1])
plt.title('Block x1 colored by y[:,0]')
plt.subplot(312)
plotdata(x2, y[:, 0:1])
plt.title('Block x2 colored by y[:,0]')
plt.subplot(313)
plotdata(x3, y[:, 0:1])
plt.title('Block x3 colored by y[:,0]')
plt.tight_layout()
plt.show()

# Here follows the calculation

pls_own = MBPLS(n_components=2)

pls_own.fit([x1, x2, x3], y[:, 0:1])

# Specify here which Component loadings and scores to plot below
pls_own.plot(1)
