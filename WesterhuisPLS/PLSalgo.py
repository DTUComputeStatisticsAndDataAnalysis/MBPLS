#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:05:52 2018

- PLS algorithm according to Westerhuis el al. 1998
- PLS is benchmarked to ScikitLearn 0.19.0 PLS algo
- data from Pectin paper (Baum et al. 2017)

@author: andba@dtu.dk
"""
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

def snv(spectra):
    stdev = np.matrix(spectra.std(axis=1)).T
    spectralmean = np.matrix(spectra.mean(axis=1)).T
    spectra_snv = (spectra-spectralmean)/stdev
    return spectra_snv


def loaddata():
    from scipy.io import loadmat
    from pylab import log10
    path = '/home/andba/Documents/Projects/DTU Compute/Novozymes BigData/MBPLS Package development/WesterhuisPLS/'
    data = loadmat(path+'data.mat')
    spectra = -log10(data['spectra_new'].T[:,18:160])
    spectra_snv = snv(spectra)
    yields = data['yield_new']
    return spectra_snv, yields


def plotdata(spectra, yields):
    plt.figure()
    for spectrum, yield1 in zip(spectra,yields/yields.max()):
        plt.plot(spectrum.T,color=[0,0.3,yield1],linewidth=2)


def preprocess(spectra,yields):
    from sklearn.preprocessing import StandardScaler
    Scaler1 = StandardScaler(with_mean=True,with_std=False)
    Scaler2 = StandardScaler(with_mean=True,with_std=True)
    return Scaler1.fit_transform(spectra), Scaler2.fit_transform(yields) 


def ScikitPLSversion(spectra_preprocessed, yields_preprocessed):
    from sklearn.cross_decomposition import PLSRegression
    ScikitPLS = PLSRegression(n_components=2, scale=False)
    ScikitPLS.fit_transform(spectra_preprocessed, yields_preprocessed)
    Scikitloadings = ScikitPLS.x_loadings_
    Scikitscores = ScikitPLS.x_scores_
    return Scikitloadings, Scikitscores


def WesterhuisPLS(X, y):
    from numpy.linalg import norm
    from numpy import dot  
    u = np.random.rand(yields.shape[0],1)   #initialize u randomly (actually for one dim y: u = y)
    ws = []
    con = 1
    counter = 1
    con_criterium = 1e-10
    max_it = 1000
    while con > con_criterium and counter < max_it:
        w = dot(X.T, u) / dot(u.T, u)   # find weights using either random u or initialized by SVD (for two dim Y)
        w_norm = w / norm(w)    # normalize w to length 1
        t = dot(X, w_norm) / dot(w.T, w_norm)
        q = dot(y.T, t) / dot(t.T, t)
        u = dot(y, q)
        ws.append(w_norm)
        if len(ws) > 1: con = sum(abs(ws[len(ws) - 1] - ws[len(ws) - 2]))
        print(con)
        counter += 1

    return ws, t, w_norm, con


spectra, yields = loaddata()
plotdata(spectra, yields)
spectra_preprocessed, yields_preprocessed = preprocess(spectra, yields)
Scikitloadings, Scikitscores = ScikitPLSversion(spectra_preprocessed, yields_preprocessed)
#plotdata(Scikitloadings.T,np.array([0,1]))
ws, t, w_norm, con = WesterhuisPLS(spectra_preprocessed, yields_preprocessed)







