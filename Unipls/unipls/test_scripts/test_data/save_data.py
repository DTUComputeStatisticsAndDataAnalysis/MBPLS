#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:56:59 2018

Use this script only to create new reference data (csv files) for post installation testing

@author: Andreas Baum, andba@dtu.dk
"""
# define Parameters
rand_seed = 25
num_samples = 20
num_vars_x1 = 25
num_vars_x2 = 45
noise = 5      # add noise between 0..10

# initialize
import numpy as np

# Generate Loadings
np.random.seed(rand_seed)
loading1 = np.expand_dims(np.random.randint(0, 10, num_vars_x1), 1)
loading2 = np.expand_dims(np.sin(np.linspace(0, 5, num_vars_x2)), 1)

# Generate orthogonal Scores
from scipy.stats import ortho_group
y = ortho_group.rvs(num_samples, random_state=rand_seed)[:, :2]

# Generate data from scores and loadings
x1 = np.dot(y[:, 0:1], loading1.T)
x2 = np.dot(y[:, 1:2], loading2.T)

# Add noise to x1 and x2 (orthogonality of Latent Variable structure will be destroyed)
x1 = np.random.normal(x1, 0.05*noise)
x2 = np.random.normal(x2, 0.05*noise)

#%% Fit MBPLS model and assert that result matches reference result
# Atm SIMPLS yields most different results; Scores and Loadings differ slightly between methods also
from unipls.mbpls import MBPLS
mbpls_model = MBPLS(n_components=2,method='UNIPALS',standardize=False)
mbpls_model.fit([x1, x2], y)
    

#%% Save results as reference data for installation testing   
from numpy import savetxt, concatenate

U = mbpls_model.U
V = mbpls_model.V
Ts = mbpls_model.Ts
T = mbpls_model.T
P = mbpls_model.P
A = mbpls_model.A
P1 = P[:num_vars_x1, :]
P2 = P[num_vars_x1:, :]
beta = mbpls_model.beta
 
savetxt('T.csv', concatenate(T,axis=1), delimiter=',')
savetxt('P1.csv', P1, delimiter=',')
savetxt('P2.csv', P2, delimiter=',')
savetxt('Ts.csv', Ts, delimiter=',')
savetxt('U.csv', U, delimiter=',')
savetxt('V.csv', V, delimiter=',')
savetxt('beta.csv', beta, delimiter=',')
savetxt('A.csv', A, delimiter=',')

