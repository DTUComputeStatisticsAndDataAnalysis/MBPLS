#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19th 12:57:33 2018

Run ade4 MBPLS for comparison

@author: andba@dtu.dk
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
num_comp = 3 

def loaddata():
    from scipy.io import loadmat
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


plt.close('all')
x1, x2, y = loaddata()
x1_process, y_process = preprocess(x1, y)
x2_process, y_process = preprocess(x2, y)

xblocks = [x1,x2]
x1_process = pd.DataFrame(x1_process)
x2_process = pd.DataFrame(x2_process)
y_process = pd.DataFrame(y_process[:,0:1])



#%%
from rpy2 import robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as rpyn

pandas2ri.activate()    # to activate easy conversion from r to pandas dataframes

# to generate a dataframe in R global environment
robjects.globalenv['x1'] = pandas2ri.py2ri_pandasdataframe(x1_process)
robjects.globalenv['x2'] = pandas2ri.py2ri_pandasdataframe(x2_process)
robjects.globalenv['ref'] = pandas2ri.py2ri_pandasdataframe(y_process)

# how to execute R code using the global environment variables defined above
robjects.r(
'''
library(ade4)
library(adegraphics)

dudiY.act <- dudi.pca(ref, center = FALSE, scale = FALSE, scannf =
                          FALSE,nf=1)
ktabX.act <- ktab.list.df(list(block1=x1,block2=x2))
resmbpls.act <- mbpls(dudiY.act, ktabX.act, scale = FALSE,
                        option = "none", scannf = FALSE, nf='''+str(num_comp)+''')
summary(resmbpls.act)
#if(adegraphicsLoaded())
#  plot(resmbpls.act)

xycoeff <- resmbpls.act$XYcoef$activity
tlx <- resmbpls.act$TlX
fax <- resmbpls.act$faX
bip <- resmbpls.act$bip
cw <- as.data.frame(resmbpls.act$X.cw)
bipc <- resmbpls.act$bipc
vip <- resmbpls.act$vip
vipc <- resmbpls.act$vipc
cov2 <- resmbpls.act$cov2
ycov <- resmbpls.act$Yco
tc1 <- resmbpls.act$Tc1
lx <- resmbpls.act$lX
ly <- resmbpls.act$lY
nf <- as.matrix(resmbpls.act$nf)
lw <- as.matrix(resmbpls.act$lw)
xcw <- as.matrix(resmbpls.act$X.cw)
blo <- as.data.frame(resmbpls.act$blo)
rank <- as.data.frame(resmbpls.act$rank)
eig <- as.matrix(resmbpls.act$eig)
yc1 <- resmbpls.act$Yc1
''')
#%%
# how to extract variables from the R global environment
#xycoeff = pandas2ri.ri2py_dataframe(robjects.globalenv['xycoeff'])
tlx = pandas2ri.ri2py_dataframe(robjects.globalenv['tlx'])
fax = pandas2ri.ri2py_dataframe(robjects.globalenv['fax'])
bip = pandas2ri.ri2py_dataframe(robjects.globalenv['bip'])
bip = pd.DataFrame(np.array(bip),index=['x1','x2'])
bipc = pandas2ri.ri2py_dataframe(robjects.globalenv['bipc'])
bipc = pd.DataFrame(np.array(bipc),index=['x1','x2'])
vip = pandas2ri.ri2py_dataframe(robjects.globalenv['vip'])
vipc = pandas2ri.ri2py_dataframe(robjects.globalenv['vipc'])
cov2 = pandas2ri.ri2py_dataframe(robjects.globalenv['cov2'])
ycov = pandas2ri.ri2py_dataframe(robjects.globalenv['ycov'])
tc1 = pandas2ri.ri2py_dataframe(robjects.globalenv['tc1'])
lx = pandas2ri.ri2py_dataframe(robjects.globalenv['lx'])
ly = pandas2ri.ri2py_dataframe(robjects.globalenv['ly'])
nf = pandas2ri.ri2py_dataframe(robjects.globalenv['nf'])
lw = pandas2ri.ri2py_dataframe(robjects.globalenv['lw'])
xcw = pandas2ri.ri2py_dataframe(robjects.globalenv['xcw'])
blo = pandas2ri.ri2py_dataframe(robjects.globalenv['blo'])
rank = pandas2ri.ri2py_dataframe(robjects.globalenv['rank'])
eig = pandas2ri.ri2py_dataframe(robjects.globalenv['eig'])
yc1 = pandas2ri.ri2py_dataframe(robjects.globalenv['yc1'])
cw = pandas2ri.ri2py_dataframe(robjects.globalenv['cw'])



#%% Post-process ade4 output

