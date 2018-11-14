#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

import os
import numpy as np
from scipy.stats import ortho_group
from urllib.request import urlretrieve
import pandas as pd
import time
import sys

all = ['data_path',
       'orthogonal_data',
       'load_Intro_Data',
       'load_FTIR_Data',
       'load_CarbohydrateMicroarrays_Data']

GITHUB_DATADIR = 'https://github.com/DTUComputeStatisticsAndDataAnalysis/MBPLS/raw/master/mbpls/data'

def data_path():
    path = os.path.dirname(os.path.abspath(__file__))
    return path


def load_Intro_Data():
    file_dict = {'extraction1': [], 'extraction2': [], 'extraction3': [],
                 'ftir1': [], 'ftir2': [], 'ftir3': []}
    dir = 'MBPLS_intro'
    file_path = os.path.join(data_path(), dir)
    path_checker(file_path)
    for file in file_dict.keys():
        abs_file_path = os.path.join(file_path, '{0}.pkl'.format(file))
        if file_checker('{0}.pkl'.format(file), abs_file_path, GITHUB_DATADIR, dir) == 0:
            file_dict[file] = pd.read_pickle(abs_file_path)
        else:
            print('File {0} was not available and could not be downloaded'.format(file))

    print("Following dataset were loaded as Pandas Dataframes: \n", file_dict.keys())
    return file_dict


def load_FTIR_Data():
    file_dict = {'ftir1': [], 'ftir2': [], 'ftir3': []}
    dir = 'FTIR'
    file_path = os.path.join(data_path(), dir)
    path_checker(file_path)
    for file in file_dict.keys():
        abs_file_path = os.path.join(file_path, '{0}.pkl'.format(file))
        if file_checker('{0}.pkl'.format(file), abs_file_path, GITHUB_DATADIR, dir) == 0:
            file_dict[file] = pd.read_pickle(abs_file_path)
        else:
            print('File {0} was not available and could not be downloaded'.format(file))

    print("Following dataset were loaded as Pandas Dataframes: \n", file_dict.keys())
    return file_dict

def load_CarbohydrateMicroarrays_Data():
    file_dict = {'extraction1': [], 'extraction2': [], 'extraction3': []}
    dir = 'CarbohydrateMicroarrays'
    file_path = os.path.join(data_path(), dir)
    path_checker(file_path)
    for file in file_dict.keys():
        abs_file_path = os.path.join(file_path, '{0}.pkl'.format(file))
        if file_checker('{0}.pkl'.format(file), abs_file_path, GITHUB_DATADIR, dir) == 0:
            file_dict[file] = pd.read_pickle(abs_file_path)
        else:
            print('File {0} was not available and could not be downloaded'.format(file))

    print("Following dataset were loaded as Pandas Dataframes: \n", file_dict.keys())
    return file_dict

def path_checker(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
            print("Path {0} was successfully created to store the data".format(path))
        except:
            print("Path {0} could not be created.".format(path))
            return 1
    return 0

def file_checker(file, abs_file_path, github_dir, dir):
    if os.path.isfile(abs_file_path):
        pass
    else: #Download the file from the github repository
        url = '{0}/{1}/{2}'.format(github_dir, dir, file)
        print('File not available locally. Trying to download file {0} from github repository.\n Link: {1}'
              .format(file, url))
        try:
            urlretrieve(url, abs_file_path, reporthook)
        except:
            print("The file could not be downloaded. Please check your internet connection.")
            return 1
    return 0


# From https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    # Adding small value for duration, to avoid division by zero
    speed = int(progress_size / (1024 * duration + 0.00001))
    percent = min(int(count*block_size*100/total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed\n" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def orthogonal_data(num_of_samples = 11, params_block_one = 4, params_block_two = 4, params_block_three = 4,
                    num_of_variables_main_lin_comb = 0, num_of_batches = 1, random_state=None):
    """This function creates a dataset with three X-blocks, which are completely orthogonal
    amongst each other and one Y-block, that has two response variables, which are a linear combination of
    the variables defined for the three blocks.

    Parameters:
    --------------
    num_of_samples: Amount of samples for the dataset
    params_block_one: Number of variables in the first block
    params_block_two: Number of variables in the second block
    params_block_three: Number of variables in the third block
    num_of_variables_main_lin_comb: Number of variables that are randon linear combinations of each variable
    (Multi-Colliniearity)
    num_of_batches: Number of batches for each block (third dimension)

    Output:
    --------------
    X_1 = First X-block - Dimensionality ( num_of_samples, params_block_one*(num_of_variables_main_lin_comb+1),
    num_of_batches)
    X_2 = Second X-block - Dimensionality ( num_of_samples, params_block_two*(num_of_variables_main_lin_comb+1),
    num_of_batches)
    X_3 = Third X-block - Dimensionality ( num_of_samples, params_block_three*(num_of_variables_main_lin_comb+1),
    num_of_batches)
    Y = Y-block - Dimensionality (num_of_samples, 2, num_of_batches)
    """
    # TODO: X = y * weight_vector

    total_params = params_block_one + params_block_three + params_block_two

    assert num_of_samples >= total_params, "The amount of samples has to be equal \
    or higher than the amount of total parameters"


    # Parameters for Y block one
    ''' Constructing two different vectors that are
    linear combinations of the main orthogonal variables'''
    lin_vec_1 = np.arange(0, total_params * 0.1, 0.1)
    temp_vec = lin_vec_1[0:8]
    temp_vec[0] = 2
    temp_vec = temp_vec / np.linalg.norm(temp_vec)
    np.linalg.norm(temp_vec[0:4]) ** 2
    np.linalg.norm(temp_vec[4:9]) ** 2
    lin_vec_1[0:8] = temp_vec
    lin_vec_2 = np.arange(total_params * 0.1 - 0.1, -0.1, -0.1)

    # Constructing X blocks

    ## X1
    if num_of_batches == 1:
        X = np.dstack(ortho_group.rvs(num_of_samples, num_of_batches, random_state=random_state))\
                                                        .transpose((0, 2, 1))[:, :, 0:total_params]
    else:
        X = ortho_group.rvs(num_of_samples, num_of_batches, random_state=random_state)\
                                                .transpose((0, 2, 1))[:, :, 0:total_params]

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
    y1 = np.einsum('j,klj->kl', lin_vec_2[0:params_block_one], X[:, :, 0:params_block_one])
    y2 = np.einsum('j,klj->kl', lin_vec_2[params_block_one:params_block_one+params_block_two],
                   X[:, :, params_block_one:params_block_one+params_block_two])
    y3 = np.einsum('j,klj->kl', lin_vec_2[params_block_one+params_block_two:total_params],
                   X[:, :, params_block_one+params_block_two:total_params])
    Y = np.stack((y1, y2, y3), -1)
    if num_of_batches == 1:
        return np.squeeze(X_1_complete), np.squeeze(X_2_complete), np.squeeze(X_3_complete), np.squeeze(Y)
    else:
        return X_1_complete, X_2_complete, X_3_complete, Y

def add_noise(data, snr):
    assert(snr>0), "The signal-to-noise-ration has to be higher than 0"
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data)
    snr_inverse = 1 / snr
    noise = np.random.randn(data.shape[0], data.shape[1]) * snr_inverse
    data = data + noise * scaler.scale_
    return data