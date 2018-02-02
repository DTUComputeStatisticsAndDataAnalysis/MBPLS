#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:31:48 2018

data from University of Copenhagen, Frans van den Berg
http://www.models.life.ku.dk/~courses/MBtoolbox/mbtmain.htm

info from data origin:
X (objects x all X-block variables) one, augmented data matrix with all the X-blocks
Xin (number of X-blocks x 2 = first and last variable per block) index for X-blocks
(optional:)
Xpp (number of X-blocks x 3) processing per X-block
 - first entry scaling: 0=do not scale to sum-of-squares, number=scale to sum-of-squares
 - second entry preprocessing: 0=none, 1=mean centering, 2=auto scaling
 - third entry = number of factors to compute for this X-block
Y (objects x all Y-block variables) one, augmented data matrix with all the Y-blocks
Yin (number of Y-blocks x 2 = first and last variable per block) index for Y-blocks
 - if omitted, one Y-block is assumed
Ypp (number of Y-blocks x 1) preprocessing per Y-block: 0=none, 1=mean centering, 2=auto scaling
Model (1 x 2) first entry : 0 = no cross validation (default), 1 = cross validation; 
  second entry overrides default modeling methods (MBCPCA for X, MBSPLS for X and Y)
- 0=MBCPCA, 1=MBPCA, 2=MBHPCA
- 10=MBSPLS, 11=MBBPLS, 12=MBHPLS

@author: Andreas Baum, andba@dtu.dk
"""

from matplotlib import pyplot as plt
import numpy as np

def loaddata():
    from scipy.io import loadmat
    from pylab import log10
    path = '/home/andba/Documents/Projects/DTU Compute/Novozymes BigData/MBPLS Package development/MBPLS/'
    data = loadmat(path+'MBdata.mat')
    y = data['Y']
    x1 = data['X'][:,:50]
    x2 = data['X'][:,50:]
    return x1, x2, y


def plotdata(spectra, yields):
    yields = yields + abs(yields.min())
    for spectrum, yield1 in zip(spectra,yields/yields.max()):
        plt.plot(spectrum.T,color=[0,0,yield1],linewidth=6)


def preprocess(spectra,yields):
    from sklearn.preprocessing import StandardScaler
    Scaler1 = StandardScaler(with_mean=True,with_std=False)
    Scaler2 = StandardScaler(with_mean=True,with_std=True)
    return Scaler1.fit_transform(spectra), Scaler2.fit_transform(yields) 

def mbpls(X, Y, n_components, full_svd=True):
    """Multiblock PLS regression with nomenclature as described in Bougeard et al 2001 (R package: ade4 mbpls function authors)
    
    ----------
    
    X : list 
        of all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
    Y : array
        1-dim or 2-dim array of reference values
    n_components : int
        Number of Latent Variables.
    
    """
    
    """
    X side:
    ts - super scores
    t - list of block scores (number of elements = number of blocks)
    w - list of block loadings (number of elements = number of blocks)
    a - list of super weights/loadings (number of elements = number of blocks)
    eigenv - normal PLS loading (of length 1)
    
    Y side:
    u - scores on Y
    eigenvy - loadings on Y
    
    """
    import numpy as np
    num_blocks = len(X)
    
    # Store start/end feature indices for all x blocks
    feature_indices = []
    for block in X:
        if len(feature_indices) > 0: 
            feature_indices.append(np.array([feature_indices[-1][1], feature_indices[-1][1] + block.shape[1]]))
        else:
            feature_indices.append(np.array([0,block.shape[1]]))
            
    # 1. Concatenate all x blocks
    Xblocks = X[:]
    X = np.hstack(X)  
    
    # 2. Calculate eigenv (normal pls loading) by SVD(X.T*Y*Y.T*X) --> eigenvector with largest eigenvalue
    S = np.dot(np.dot(np.dot(X.T,Y),Y.T),X)
    eigenv = np.linalg.svd(S, full_matrices=full_svd)[0][:,0:1]

    # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ... 
    w = []
    a = []
    for indices, block in zip(feature_indices, range(num_blocks)):
        partialloading = eigenv[indices[0]:indices[1]]
        w.append(partialloading / np.linalg.norm(partialloading))
        a.append(np.linalg.norm(partialloading))

    # 4. Calculate block scores t1, t2, ... as tn = Xn*wn
    t = []
    for block, blockloading in zip(Xblocks, w):
        t.append(np.dot(block, blockloading))
    
    # 5. Calculate super scores ts
    ts = np.dot(X, eigenv)
    
    # 6. Calculate eigenvy by SVD(Y.T*X*X.T*Y)
    S = np.dot(np.dot(np.dot(Y.T,X),X.T),Y)
    eigenvy = np.linalg.svd(S, full_matrices=full_svd)[0][:,0:1]
    
    # 7. Calculate scores u 
    u = np.dot(Y, eigenvy)

    # 8. Deflate X by calculating: Xnew = X - ts*eigenv.T
    Xnew = X - np.dot(ts, eigenv.T)
    
    # 9. Deflate Y by calculating: Ynew = Y - u*eigenvy.T
    Ynew = Y - np.dot(u, eigenvy.T)
    
    return eigenv, eigenvy, w, a, t, ts
        
    
plt.close('all')
x1, x2, y = loaddata()
x1_process, y_process = preprocess(x1, y)
x2_process, y_process = preprocess(x2, y)

# Plot preprocessed datablocks X1 and X2 colored according to y1, y2, y3
plt.subplot(231)
plotdata(x1, y[:,0:1]); plt.title('Block x1 colored by y[:,0]')
plt.subplot(232)
plotdata(x1, y[:,1:2]); plt.title('Block x1 colored by y[:,1]')
plt.subplot(233)
plotdata(x1, y[:,2:3]); plt.title('Block x1 colored by y[:,2]')

plt.subplot(234)
plotdata(x2, y[:,0:1]); plt.title('Block x2 colored by y[:,0]')
plt.subplot(235)
plotdata(x2, y[:,1:2]); plt.title('Block x2 colored by y[:,1]')
plt.subplot(236)
plotdata(x2, y[:,2:3]); plt.title('Block x2 colored by y[:,2]')

eigenv, eigenvy, w, a, t, ts = mbpls([x1_process, x2_process], y_process, n_components=1)

plt.figure()
plt.subplot(221)
plt.plot(w[0]); plt.title('block loading x1\nBlock importance: ' + str(a[0]**2))
plt.subplot(222)
plt.plot(w[1]); plt.title('block loading x2\nBlock importance: ' + str(a[1]**2))
plt.subplot(223)
plt.plot(t[0]); plt.title('block scores x1')
plt.subplot(224)
plt.plot(t[1]); plt.title('block scores x2')
