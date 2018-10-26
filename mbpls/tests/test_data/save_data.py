#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:56:59 2018

Use this script only to create new reference data (csv files) for post installation testing
- two sections for cases of p > n and p < n

@author: Andreas Baum, andba@dtu.dk
"""
#%% define Parameters fror case of p > n
rand_seed = 25
num_samples = 50
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

# Separate Data into training and test sets
indices = np.random.choice(np.arange(num_samples),num_samples,replace=False)
train, test = indices[:round(num_samples*8/10)], indices[round(num_samples*8/10):]
x1_train, x2_train = x1[train,:], x2[train,:]
x1_test, x2_test = x1[test,:], x2[test,:]
y_train, y_test = y[train,:], y[test,:]

# Fit MBPLS model, transform and predict and save results as reference
# SIMPLS doesn't give Multiblock results, therefore left out for A (block importance) and T (block scores)
from mbpls.mbpls import MBPLS
from numpy import savetxt, concatenate
methods = ['UNIPALS', 'NIPALS', 'KERNEL', 'SIMPLS']
for method in methods:
    # Fit MBPLS models
    mbpls_model = MBPLS(n_components=2,method=method,standardize=True)
    mbpls_model.fit([x1_train, x2_train], y_train)
    U = mbpls_model.U_
    V = mbpls_model.V_
    Ts = mbpls_model.Ts_
    P = mbpls_model.P_
    P1 = P[0]
    P2 = P[1]
    beta = mbpls_model.beta_
    if method is not 'SIMPLS': 
        T = mbpls_model.T_
        A = mbpls_model.A_
        savetxt('T_%s.csv' % method, concatenate(T, axis=1), delimiter=',')
        savetxt('A_%s.csv' % method, A, delimiter=',')
    savetxt('P1_%s.csv' % method, P1, delimiter=',')
    savetxt('P2_%s.csv' % method, P2, delimiter=',')
    savetxt('Ts_%s.csv' % method, Ts, delimiter=',')
    savetxt('U_%s.csv' % method, U, delimiter=',')
    savetxt('V_%s.csv' % method, V, delimiter=',')
    savetxt('beta_%s.csv' % method, beta, delimiter=',')
    
    # Transform test data using MBPLS model
    if method is not 'SIMPLS':
        Ts_test, T_test, U_test = mbpls_model.transform([x1_test, x2_test], y_test)
        savetxt('T_test_%s.csv' % method, concatenate(T_test, axis=1), delimiter=',')
    else:
        Ts_test, U_test = mbpls_model.transform([x1_test, x2_test], y_test)
    savetxt('Ts_test_%s.csv' % method, Ts_test, delimiter=',')
    savetxt('U_test_%s.csv' % method, U_test, delimiter=',')
    
    # Predict y_test using MBPLS model
    y_predict = mbpls_model.predict([x1_test, x2_test])
    savetxt('Y_predict_test_%s.csv' % method, y_predict, delimiter=',')


#%% define Parameters fror case of n > p
rand_seed = 25
num_samples = 150
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

# Separate Data into training and test sets
indices = np.random.choice(np.arange(num_samples),num_samples,replace=False)
train, test = indices[:round(num_samples*8/10)], indices[round(num_samples*8/10):]
x1_train, x2_train = x1[train,:], x2[train,:]
x1_test, x2_test = x1[test,:], x2[test,:]
y_train, y_test = y[train,:], y[test,:]


# Fit MBPLS model, transform and predict and save results as reference (for transformed matrix X)
from mbpls.mbpls import MBPLS
from numpy import savetxt, concatenate
methods = ['UNIPALS', 'KERNEL']
for method in methods:
    # Fit MBPLS models
    mbpls_model = MBPLS(n_components=2,method=method,standardize=True)
    mbpls_model.fit([x1_train, x2_train], y_train)
    U = mbpls_model.U_
    V = mbpls_model.V_
    Ts = mbpls_model.Ts_
    P = mbpls_model.P_
    P1 = P[0]
    P2 = P[1]
    beta = mbpls_model.beta_
    if method is not 'SIMPLS': 
        T = mbpls_model.T_
        A = mbpls_model.A_
        savetxt('T_NlargerP_%s.csv' % method, concatenate(T, axis=1), delimiter=',')
        savetxt('A_NlargerP_%s.csv' % method, A, delimiter=',')
    savetxt('P1_NlargerP_%s.csv' % method, P1, delimiter=',')
    savetxt('P2_NlargerP_%s.csv' % method, P2, delimiter=',')
    savetxt('Ts_NlargerP_%s.csv' % method, Ts, delimiter=',')
    savetxt('U_NlargerP_%s.csv' % method, U, delimiter=',')
    savetxt('V_NlargerP_%s.csv' % method, V, delimiter=',')
    savetxt('beta_NlargerP_%s.csv' % method, beta, delimiter=',')
    
    # Transform test data using MBPLS model
    if method is not 'SIMPLS':
        Ts_test, T_test, U_test = mbpls_model.transform([x1_test, x2_test], y_test)
        savetxt('T_test_NlargerP_%s.csv' % method, concatenate(T_test, axis=1), delimiter=',')
    else:
        Ts_test, U_test = mbpls_model.transform([x1_test, x2_test], y_test)
    savetxt('Ts_test_NlargerP_%s.csv' % method, Ts_test, delimiter=',')
    savetxt('U_test_NlargerP_%s.csv' % method, U_test, delimiter=',')
    
    # Predict y_test using MBPLS model
    y_predict = mbpls_model.predict([x1_test, x2_test])
    savetxt('Y_predict_test_NlargerP_%s.csv' % method, y_predict, delimiter=',')

