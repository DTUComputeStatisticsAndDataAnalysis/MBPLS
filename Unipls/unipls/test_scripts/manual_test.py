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
@edited: Laurent Vermue, lauve@dtu.dk
"""
#%%
from matplotlib import pyplot as plt
from unipls.data.get_data import data_path
from unipls.mbpls import MBPLS
import numpy as np
import os

plt.ioff()


def loaddata():
    from scipy.io import loadmat
    data = loadmat(os.path.join(data_path(), 'MBdata.mat'))
    y = data['Y']
    x1 = data['X'][:, :50]
    x2 = data['X'][:, 50:]
    return x1, x2, y


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
x1, x2, y = loaddata()
x1_process, y_process = preprocess(x1, y)
x2_process, y_process = preprocess(x2, y)

## Plot preprocessed datablocks X1 and X2 colored according to y1, y2, y3
#plt.subplot(231)
#plotdata(x1, y[:, 0:1])
#plt.title('Block x1 colored by y[:,0]')
#plt.subplot(232)
#plotdata(x1, y[:, 1:2])
#plt.title('Block x1 colored by y[:,1]')
#plt.subplot(233)
#plotdata(x1, y[:, 2:3])
#plt.title('Block x1 colored by y[:,2]')
#
#plt.subplot(234)
#plotdata(x2, y[:, 0:1])
#plt.title('Block x2 colored by y[:,0]')
#plt.subplot(235)
#plotdata(x2, y[:, 1:2])
#plt.title('Block x2 colored by y[:,1]')
#plt.subplot(236)
#plotdata(x2, y[:, 2:3])
#plt.title('Block x2 colored by y[:, 2]')
#plt.show()

# Here follows the calculation

UNI = MBPLS(n_components=5, method='UNIPALS', standardize=True)
NIPALS = MBPLS(n_components=5, method='NIPALS', standardize=True)
SIMPLS = MBPLS(n_components=5, method='SIMPLS', standardize=True)
KERNEL = MBPLS(n_components=5, method='KERNEL', standardize=True)

UNI.fit([x1, x2], y[:, 0:1])
NIPALS.fit([x1, x2], y[:, 0:1])
SIMPLS.fit([x1, x2], y[:, 0:1])
KERNEL.fit([x1, x2], y[:, 0:1])
#KERNEL.fit([x1.T, x2.T], np.repeat(y[:, 0:1], 5, axis=0))
# Calculate normalized weights in one shot
#(x1_new.T.dot(KERNEL.U[:,1:2])/np.linalg.norm(x1_new.T.dot(KERNEL.U[:,1:2])))[:10]
#x1_new = UNI.x_scalers[0].transform(x1)

##### Testing



#%%

# Other way to calculate block importances
var=5
test = SIMPLS.W[:, var-1:var]
test = test / np.linalg.norm(test)
np.linalg.norm(test[:50]) ** 2

test = np.concatenate((UNI.W[0][:, 1:2],  UNI.W[1][:, 1:2]))
np.linalg.norm(test[:50]) ** 2

# Probably correct manner to calculate Block importance
p=SIMPLS.P[:, 1:2]
t=SIMPLS.Ts[:,1:2]
u = SIMPLS.U[:, 1:2]
w = p.dot(np.linalg.pinv(t)).dot(u)
w = w / np.linalg.norm(w)
np.linalg.norm(w[:50])**2

# Proof of block importance being calculated this way (Based on loadings)
# ts = T * superweight / superweight.T * superweight
Ts = np.hstack((UNI.T[0][:,1:2], UNI.T[1][:, 1:2]))
ts = UNI.Ts[:, 1:2]
w = Ts.T.dot(np.linalg.pinv(ts).T)
w = w / np.linalg.norm(w)
w ** 2

#np.cov(SIMPLS.Ts[:, 1:2].T, SIMPLS.U[:, 1:2].T)
#%%
print(np.sum(np.abs(SIMPLS.Ts)-np.abs(UNI.Ts)))
X=np.hstack((x1, x2))
X.dot(UNI.beta)



#%% Testing on data from above

NIPALS = MBPLS(n_components=2, method='NIPALS', standardize=True)
NIPALS.fit([x1, x2], y[:, 0:2])
print(NIPALS.A)

NIPALS = MBPLS(n_components=2, method='NIPALS', standardize=False)
NIPALS.fit([x1, x2], y[:, 0:2])
print(NIPALS.A)

#%% Testing vectors that are not completely othogonal
a=np.random.randint(1, 20, size=(2000, 500))
y_a=a[:, 0:1] * 100
y_a = y_a.T + np.random.rand(a[:, 0:1].shape[0])
y_a = y_a.T
a1=a[:, 0:2]
a2=a[:, 2:]

# Not standardizing the blocks renders the blockimportance useless

NIPALS = MBPLS(n_components = 2, method='NIPALS', standardize=True)
NIPALS.fit([a1, a2], y_a)
print(NIPALS.A)

NIPALS = MBPLS(n_components = 2, method='NIPALS', standardize=False)
NIPALS.fit([a1, a2], y_a)
print(NIPALS.A)

#%% Testing vectors that are not completely othogonal
a=np.random.randint(1, 20, size=(2000, 500))
y_a=a[:, 0:1] * 100
y_a = y_a.T + np.random.rand(a[:, 0:1].shape[0])
y_a = y_a.T
a1=a[:, 0:2]
a2=a[:, 2:]

# Not standardizing the blocks renders the blockimportance useless

UNI = MBPLS(n_components = 2, method='UNIPALS', standardize=True)
UNI.fit([a1, a2], y_a)
print(UNI.A)

UNI = MBPLS(n_components = 2, method='UNIPALS', standardize=False)
UNI.fit([a1, a2], y_a)
print(UNI.A)

#%% Testing completely orthogonal vectors
from scipy.stats import ortho_group
A = ortho_group.rvs(100)
y_a=A[:, 0:1] * 1000 + A[:, 1:2] * 100
y_a = y_a.T + np.random.rand(A[:, 0:1].shape[0])
y_a = y_a.T
a1=A[:, 0:2]
a2=A[:, 2:]

# Not standardizing the blocks renders the blockimportance useless

NIPALS = MBPLS(n_components = 2, method='SIMPLS', standardize=True)
NIPALS.fit([a1, a2], y_a)
print(NIPALS.A)
NIPALS = MBPLS(n_components = 2, method='SIMPLS', standardize=False)
NIPALS.fit([a1, a2], y_a)
print(NIPALS.A)


#%%

# Specify here which Component loadings and scores to plot below
UNI.plot(4)
# SVD.plot([1,3,5])
NIPALS.plot(4)

# Scikit Learn PLS
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=2, scale=False)
scikitpls = pls.fit(X=np.hstack((x1_process, x2_process)), Y=y_process[:, 0:1])
#scikitscores = scikitpls.x_scores_
#scikitloadings = scikitpls.x_loadings_

#pls = PLSSVD(n_components=1, scale=False)
#scikitpls = pls.fit(X=np.hstack((x1_process, x2_process)), Y=y_process[:,0:1])
#scikitscores = scikitpls.x_scores_

#print(NIPALS.W[1][10,1])
#print(SVD.W[1][10,1])

#%% Testing
UNI = MBPLS(n_components=5, method='UNIPALS', standardize=False)
UNI.fit([x1, x2], y[:, 0:1])
UNI.plot(4)