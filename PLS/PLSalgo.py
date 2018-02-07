#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:05:52 2018

- NIPALS PLS according to Westerhuis el al. 1998 and SIMPLS PLS according to Jong 1992
- 2nd SIMPLS PLS algorithm according to ade4 MBPLS paper
- PLS results are benchmarked to ScikitLearn 0.19.0 PLS algo
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
    path = '/home/andba/Documents/Projects/DTU Compute/Novozymes BigData/MBPLS Package development/PLS/'
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


def NIPALSPLS(X, y, num_comp):
    from numpy.linalg import norm
    from numpy import dot, empty, hstack  
    u = np.random.rand(yields.shape[0],1)   # initialize u randomly (actually for one dim u = y)
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
            w_norm = w / norm(w)
            t = dot(X, w_norm) / dot(w_norm.T, w_norm)
            q = dot(y.T, t) / dot(t.T, t)   
            u = dot(y, q)
            ts.append(t)
            if len(ts) > 1: con = sum(abs(ts[len(ts) - 1] - ts[len(ts) - 2]))
            counter += 1
        
        p = dot(X.T, t) / dot(t.T, t)       # x loading
        X = X - dot(t, p.T)                 # deflate X
        y = y - dot(t, q.T)                 # deflate y
        loadings = hstack((loadings, p))
        scores = hstack((scores, t))
    return loadings, scores


def SIMPLSpls(X, y, num_comp):              # orthogonlization step is missing at the moment
    from numpy.linalg import norm, svd
    from numpy import dot, empty, hstack, expand_dims
    loadings = empty((X.shape[1],0))
    scores = empty((X.shape[0],0))
    rr = empty((X.shape[1],0))
    qq = empty((y.shape[1],0))

    s = dot(X.T, y)
    for comp in range(num_comp):
        q = svd(dot(s.T, s))[1]
        q = expand_dims(q, 1)
        r = dot(s, q)
        r = r / norm(r)
        t = dot(X, r)
        #t = t / norm(t)
        p = dot(X.T, t)
        p = p / norm(p)                         # actually performed after orthogonalization step
        s = s - dot(p, dot(p.T, s))
        loadings = hstack((loadings, p))
        scores = hstack((scores, t))
        rr = hstack((rr, r))
        qq = hstack((qq, q))
    
    b = dot(rr, qq.T)
    return loadings, scores, b


def SIMPLSpls_ade4(X, y, num_comp):              
    from numpy.linalg import norm, svd
    from numpy import dot, empty, hstack
    xloadings = empty((X.shape[1],0))
    xscores = empty((X.shape[0],0))
    yloadings = empty((y.shape[1],0))
    yscores = empty((y.shape[0],0))

    for comp in range(num_comp):
        # Xside
        s = dot(dot(dot(X.T,y),y.T),X)
        p = svd(s)[0][:,0:1]
        t = dot(X,p)
        xloadings = hstack((xloadings, p))
        xscores = hstack((xscores, t))
        # Yside (as projection of t on y)
        v = dot(y.T,t)     
        u = dot(y,v)
        yloadings = hstack((yloadings, v))
        yscores = hstack((yscores, u))
        # deflate on X and Y
        X = X - dot(t,p.T)
        y = y - dot(t,v.T) 
    
    return xloadings, xscores, yloadings, yscores

spectra, yields = loaddata()
plotdata(spectra, yields)
plt.title('Spectra colored according to target yield')
spectra_preprocessed, yields_preprocessed = preprocess(spectra, yields)

plt.figure()
Scikitloadings, Scikitscores = ScikitPLSversion(spectra_preprocessed, yields_preprocessed)
plotdata(Scikitloadings.T, np.array([0,0]))
plt.title('Scikit Learn PLS loadings')

plt.figure()
NIPALSloadings, NIPALSscores = NIPALSPLS(spectra_preprocessed, yields_preprocessed, num_comp=2)
plotdata(NIPALSloadings.T, np.array([1,1]))
plt.title('NIPALS PLS loadings')

plt.figure()
SIMPLSloadings, SIMPLSscores, SIMPLSb = SIMPLSpls(spectra_preprocessed, yields_preprocessed, num_comp=2)
plotdata(SIMPLSloadings.T, np.array([0.5, 0.5]))
plt.title('deJong SIMPLS PLS loadings')

plt.figure()
SIMPLSade4loadings, SIMPLSade4scores, ade4yloadings, ade4yscores = SIMPLSpls_ade4(spectra_preprocessed, yields_preprocessed, num_comp=2)
plotdata(SIMPLSade4loadings.T, np.array([0.5, 0.5]))
plt.title('ade4 SIMPLS PLS loadings')





