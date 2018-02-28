#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# author: Laurent Vermue
# author_email: lauve@dtu.dk
#

import numpy as np
from scipy.stats import norm
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

# General parameters for all blocks
num_of_batches = 1

# Parameters for X
num_of_variables_main_lin_comb = 9
num_of_samples = 11 # This value has to be higher than the amount of total_params
params_block_one = 4
params_block_two = 4
params_block_three = 3
total_params = params_block_one + params_block_three + params_block_two


assert num_of_samples >= total_params, "The amount of samples has to be equal \
or higher than the amount of total parameters"


# Parameters for Y block one
''' Constructing two different vectors that are
linear combinations of the main orthogonal variables'''
lin_vec_1 = np.arange(0, total_params * 0.1, 0.1)
lin_vec_2 = np.arange(total_params * 0.1 - 0.1, -0.1, -0.1)

# Constructing X blocks

## X1
if num_of_batches == 1:
    X = np.dstack(ortho_group.rvs(num_of_samples, num_of_batches)).transpose((0, 2, 1))[:, :, 0:total_params]
else:
    X = ortho_group.rvs(num_of_samples, num_of_batches).transpose((0,2,1))[:, :, 0:total_params]

X_1 = X[:, :, 0:params_block_one]
# Adding linear combinations
rand_linear_factors = np.random.rand(num_of_variables_main_lin_comb)
X_1_linear_comb = np.einsum('ijk,l->ijkl', X_1, rand_linear_factors).reshape((num_of_batches, num_of_samples, -1))
X_1_complete = np.concatenate((X_1, X_1_linear_comb), -1)

## X2
X_2 = X[:, :, params_block_one:params_block_one+params_block_two]
# Adding linear combinations
rand_linear_factors = np.random.rand(num_of_variables_main_lin_comb)
X_2_linear_comb = np.einsum('ijk,l->ijkl', X_2, rand_linear_factors).reshape((num_of_batches, num_of_samples, -1))
X_2_complete = np.concatenate((X_2, X_2_linear_comb), -1)

## X3
X_3 = X[:, :, params_block_one+params_block_two:total_params]
# Adding linear combinations
rand_linear_factors = np.random.rand(num_of_variables_main_lin_comb)
X_3_linear_comb = np.einsum('ijk,l->ijkl', X_3, rand_linear_factors).reshape((num_of_batches, num_of_samples, -1))
X_3_complete = np.concatenate((X_3, X_3_linear_comb), -1)

# Constructing Y block
y1 = np.einsum('j,klj->kl', lin_vec_1, X[:, :, 0:total_params])
y2 = np.einsum('j,klj->kl', lin_vec_2, X[:, :, 0:total_params])
Y = np.stack((y1, y2), -1)