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


def WesterhuisPLS(X, y, num_comp):
    from numpy.linalg import norm
    from numpy import dot, empty, hstack  
    u = np.random.rand(yields.shape[0],1)   # initialize u randomly (actually for one dim y: u = y)
    loadings = empty((X.shape[1],0))
    scores = empty((X.shape[0],0))
    con_criterium = 1e-20
    max_it = 1000
    for comp in range(num_comp):
        ts = []
        con = 1
        counter = 1
        while con > con_criterium and counter < max_it:
            w = dot(X.T, u) / dot(u.T, u)   # find x weights using either random u or initialized by SVD (for two dim Y)
            w_norm = w / norm(w)            # normalize w to length 1
            t = dot(X, w_norm) / dot(w_norm.T, w_norm)
            q = dot(y.T, t) / dot(t.T, t)   # find y weights
            u = dot(y, q)
            ts.append(t)
            if len(ts) > 1: con = sum(abs(ts[len(ts) - 1] - ts[len(ts) - 2]))
            counter += 1
            print(con)
        
        p = dot(X.T, t) / dot(t.T, t)       # x loading
        X = X - dot(t, p.T)                 # deflate X
        y = y - dot(t, q.T)                 # deflate y
        loadings = hstack((loadings, p))
        scores = hstack((scores, t))
    return loadings, scores


spectra, yields = loaddata()
plotdata(spectra, yields)
plt.title('Spectra colored according to target yield')
spectra_preprocessed, yields_preprocessed = preprocess(spectra, yields)

plt.figure()
Scikitloadings, Scikitscores = ScikitPLSversion(spectra_preprocessed, yields_preprocessed)
plotdata(Scikitloadings.T,np.array([0,0]))
plt.title('Scikit Learn PLS loadings')

plt.figure()
Westerhuisloadings, Westerhuisscores = WesterhuisPLS(spectra_preprocessed, yields_preprocessed, num_comp=2)
plotdata(Westerhuisloadings.T,np.array([1,1]))
plt.title('Westerhuis Paper PLS loadings')





