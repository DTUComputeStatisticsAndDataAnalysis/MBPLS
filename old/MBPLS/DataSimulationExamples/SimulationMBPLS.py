#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:58:07 2018

@author: andba
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ortho_group

plt.close('all')

num_vars_x1 = 40
num_vars_x2 = 60
num_samples = 4

#construct loadings
def makesinloading(num_vars):
    temp = np.linspace(0,np.pi,num_vars)
    return np.matrix(np.sin(temp)).T

def makesin2loading(num_vars):
    temp = np.linspace(0,np.pi*2,num_vars)
    return np.matrix(np.sin(temp+0.8)).T

def makedata(loading, y):
    return np.dot(loading, y.T)

#make orthonormal scores and generate xblock data from them
var = ortho_group.rvs(num_samples)
#var = np.random.rand(30,30)   # non-orthogonal scores as reference
#var = np.array([[0,0,0,0,0,1,2,3,4,5],[5,4,3,2,1,0,0,0,0,0]]).T
y = var[:,:2]

#Loadings related to bocks x1 and x2 do not have to be orthogonal
loading1_x1 = makesinloading(num_vars_x1)
loading1_x2 = makesin2loading(num_vars_x2)
x1 = makedata(loading1_x1, y[:,0:1])
x2 = makedata(loading1_x2, y[:,1:2])

plt.figure()
plt.plot(np.concatenate((x1,x2)))


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
    Ts - super scores
    T - list of block scores (number of elements = number of blocks)
    W - list of block loadings (number of elements = number of blocks)
    A - super weights/loadings (number of rows = number of blocks)
    eigenv - normal PLS loading (of length 1)
    
    Y side:
    U - scores on Y
    V - loadings on Y
    
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
    
    W = []
    P = []
    T = []
    V = []
    U = []
    A = np.empty((num_blocks,0))
    V = np.empty((Y.shape[1],0))
    U = np.empty((Y.shape[0],0))
    Ts = np.empty((Y.shape[0],0))
    
    for block in range(num_blocks):
        W.append(np.empty((X[block].shape[1],0)))
        T.append(np.empty((X[block].shape[0],0)))
        

    # Concatenate X blocks
    X = np.hstack(X)
    
    for comp in range(n_components):
        # 1. Restore X blocks (for each deflation step)
        Xblocks = []
        for indices in feature_indices:
            Xblocks.append(X[:,indices[0]:indices[1]])
        
        # 2. Calculate eigenv (normal pls loading) by SVD(X.T*Y*Y.T*X) --> eigenvector with largest eigenvalue
        S = np.dot(np.dot(np.dot(X.T,Y),Y.T),X)
        eigenv = np.linalg.svd(S, full_matrices=full_svd)[0][:,0:1]
    
        # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ... 
        w = []
        a = []
        for indices, block in zip(feature_indices, range(num_blocks)):
            partialloading = eigenv[indices[0]:indices[1]]
            w.append(partialloading / np.linalg.norm(partialloading))
            a.append(np.linalg.norm(partialloading)**2)
    
        # 4. Calculate block scores t1, t2, ... as tn = Xn*wn
        t = []
        for block, blockloading in zip(Xblocks, w):
            t.append(np.dot(block, blockloading))
        
        # 5. Calculate super scores ts
        ts = np.dot(X, eigenv)
                
        # 6. Calculate v (Y-loading) by projection of ts on Y
        v = np.dot(Y.T, ts)
        v = v / np.linalg.norm(v)
                
        # 7. Calculate u (Y-scores) 
        u = np.dot(Y, v)
    
        # 8. Deflate X by calculating: Xnew = X - ts*loading (you need to find a loading for deflation by projecting the scores onto X)
        loading = np.dot(X.T, ts) / np.dot(ts.T, ts)
        X = X - np.dot(ts, loading.T)
                
        # 9. Deflate Y by calculating: Ynew = Y - ts*eigenvy.T (deflation on Y is optional)
        #Y = Y - np.dot(ts, v.T)
        
        # 10. add t, w, u, v, ts and a to T, W, U, V, Ts and A
        V = np.hstack((V, v))
        U = np.hstack((U, u))
        A = np.hstack((A, np.matrix(a).T))
        Ts = np.hstack((Ts, ts))
        for block in range(num_blocks):
            W[block] = np.hstack((W[block], w[block]))
            T[block] = np.hstack((T[block], t[block]))
        
        plt.plot(eigenv)
            
    return W, P, T, V, U, A, Ts

"""
#%% remove orthogonal information to y in x blocks (e.g. added noise)
which_y = 0
x = np.concatenate((x1.T,x2.T),axis=1)
corr = np.dot(np.dot(y[:,which_y:which_y+1],np.linalg.pinv(y[:,which_y:which_y+1])),x)
x = x-corr
plt.figure()
plt.plot(x.T)
x1 = x[:,0:num_vars_x1].T
x2 = x[:,num_vars_x1:num_vars_x1+num_vars_x2].T
"""

# Perform MBPLS (do not center or scale --> destroys orthogonality in Y and/or creates offset between X and Y)
W, P, T, V, U, A, Ts = mbpls([x1.T, x2.T], y[:,0:2], n_components=2)

plot_comp = 0
plt.figure()
plt.subplot(321)
plt.plot(W[0][:,plot_comp]); plt.title('block loading x1\nBlock importance: ' + str(round(A[0,plot_comp], 2)))
plt.subplot(322)
plt.plot(W[1][:,plot_comp]); plt.title('block loading x2\nBlock importance: ' + str(round(A[1,plot_comp], 2)))
plt.subplot(323)
plt.plot(T[0][:,plot_comp]); plt.title('block scores x1')
plt.subplot(324)
plt.plot(T[1][:,plot_comp]); plt.title('block scores x2')
plt.subplot(325)
plt.scatter(y[:,0], np.array(T[0][:,0])); plt.title('known y%s versus block%s scores' % (plot_comp+1, plot_comp+1))

plot_comp = 1
plt.figure()
plt.subplot(321)
plt.plot(W[0][:,plot_comp]); plt.title('block loading x1\nBlock importance: ' + str(round(A[0,plot_comp], 2)))
plt.subplot(322)
plt.plot(W[1][:,plot_comp]); plt.title('block loading x2\nBlock importance: ' + str(round(A[1,plot_comp], 2)))
plt.subplot(323)
plt.plot(T[0][:,plot_comp]); plt.title('block scores x1')
plt.subplot(324)
plt.plot(T[1][:,plot_comp]); plt.title('block scores x2')
plt.subplot(325)
plt.scatter(y[:,0], np.array(T[0][:,0])); plt.title('known y%s versus block%s scores' % (plot_comp+1, plot_comp+1))