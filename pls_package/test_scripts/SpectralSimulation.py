#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:03:12 2018

- This test script generates data using three spectral parts as input for loadings 1, 2 and 3
- The data matrix X is composed as X = TP.T
- columns in T are orthoghonal
- num_vars_x1 + num_vars_x2 must equal 40 (at current stage)
- BIPs are calculated from blocked loadings
- MBPLS is performed, results are plotted and compared to given BIPs

@author: Andreas Baum, andba@dtu.dk
"""
from matplotlib import pyplot as plt
import numpy as np
from pls_package.all_pls.mbpls import MBPLS
from pls_package.all_pls.data.get_filepath import data_path

plt.close('all')

rand_seed = 25
num_samples = 100
num_vars_x1 = 25
num_vars_x2 = 15

factors_loading1 = [1,0]
factors_loading2 = [0,1]
factors_loading3 = [0.6,0.4]

def load_loadings():
    from scipy.io import loadmat 
    data = loadmat(data_path()+'/TrametesVersicolor.mat')['samples'][0,2]
    loading1 = data[18:58,0:1] / np.linalg.norm(data[18:58,0])
    loading2 = data[80:120,0:1] / np.linalg.norm(data[200:240,0])
    loading3 = data[120:160,0:1] / np.linalg.norm(data[120:160,0])
    return loading1, loading2, loading3


def makedata(loading, y):
    return np.dot(y, loading.T)

def make_orth_scores(num_samples, rand_seed):
    from scipy.stats import ortho_group
    var = ortho_group.rvs(num_samples, random_state=rand_seed)
    return var[:,:3]

def calc_bips(loadings, num_vars_x1, num_vars_x2):
    bips = []
    for loading in loadings.T:
        bip = []
        bip.append(np.linalg.norm(loading.T[:num_vars_x1])**2)
        bip.append(np.linalg.norm(loading.T[num_vars_x1:num_vars_x1+num_vars_x2])**2)
        bips.append(bip)
    return bips

def make_blocked_loadings(loading, factors_loading):
    loading_x1 = loading[:num_vars_x1,:1]
    loading_x1 = loading_x1 * factors_loading[0] 
    loading_x2 = loading[num_vars_x1:num_vars_x1+num_vars_x2,:1]
    loading_x2 = loading_x2 * factors_loading[1] 
    return np.vstack((loading_x1,loading_x2))

#%% Generate Data using loadings in x1 and x2 (*factors) and orthogonal scores
y = make_orth_scores(num_samples, rand_seed)
loading1, loading2, loading3 = load_loadings()

plt.subplot(121)
plt.plot(np.concatenate((loading1, loading2, loading3),axis=1))
plt.xlabel('features')
plt.ylabel('arbitrary intensity')
plt.title('original spectral loadings\nprior to weighting in blocks')

loading1 = make_blocked_loadings(loading1, factors_loading1)
loading2 = make_blocked_loadings(loading2, factors_loading2)
loading3 = make_blocked_loadings(loading3, factors_loading3)

loading1 = loading1 / np.linalg.norm(loading1)
loading2 = loading2 / np.linalg.norm(loading2)
loading3 = loading3 / np.linalg.norm(loading3)

loadings = np.hstack((loading1, loading2, loading3))

bips = calc_bips(loadings, num_vars_x1, num_vars_x2)

x = makedata(loadings, y)
x1 = x[:,:num_vars_x1]
x2 = x[:,num_vars_x1:num_vars_x1+num_vars_x2]

plt.subplot(122)
plt.plot(loadings)
plt.xlabel('features')
plt.ylabel('arbitrary intensity')
plt.title('spectral loadings after weighting\nin blocks by block_factors')

plt.figure()
plt.plot(np.concatenate((x1,x2),axis=1).T, color='k')
plt.xlabel('features')
plt.ylabel('arbitrary intensity')
plt.title('Data as constructed by X = TP.T')

# Fit MBPLS model
which_y = 2
pls_own = MBPLS(n_components=1)
pls_own.fit([x1, x2], y[:,which_y-1:which_y])

# Plot MBPLS result
comp = 1
pls_own.plot(comp)

# print pre-calculated block importance
print('known BIPs: %f, %f' % (bips[which_y-1][0], bips[which_y-1][1]))

