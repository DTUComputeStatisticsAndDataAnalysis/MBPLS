"""
Script to check the availability of test files
"""

#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

import os
from ..data.get_data import file_checker, path_checker

TEST_FILES = ['P1_KERNEL.csv',
              'U_NlargerP_UNIPALS.csv',
              'Y_predict_test_NIPALS.csv',
              'Ts_test_SIMPLS.csv',
              'Ts_NIPALS.csv',
              'Ts_test_NlargerP_UNIPALS.csv',
              'U_test_NIPALS.csv',
              'beta_KERNEL.csv',
              'P2_NlargerP_UNIPALS.csv',
              'V_NIPALS.csv',
              'A_UNIPALS.csv',
              'Ts_UNIPALS.csv',
              'A_NlargerP_UNIPALS.csv',
              'P2_NIPALS.csv',
              'U_KERNEL.csv',
              'Y_predict_test_NlargerP_UNIPALS.csv',
              'T_test_UNIPALS.csv',
              'T_UNIPALS.csv',
              'T_NlargerP_UNIPALS.csv',
              'T_NIPALS.csv',
              'A_KERNEL.csv',
              'U_SIMPLS.csv',
              'Ts_NlargerP_UNIPALS.csv',
              'Ts_test_KERNEL.csv',
              'beta_NlargerP_UNIPALS.csv',
              'P1_SIMPLS.csv',
              'T_test_NlargerP_UNIPALS.csv',
              'P2_NlargerP_KERNEL.csv',
              'P1_NlargerP_KERNEL.csv',
              'T_test_NIPALS.csv',
              'U_test_UNIPALS.csv',
              'beta_SIMPLS.csv',
              'U_NlargerP_KERNEL.csv',
              'Ts_test_UNIPALS.csv',
              'V_UNIPALS.csv',
              'P2_SIMPLS.csv',
              'T_NlargerP_KERNEL.csv',
              'T_test_NlargerP_KERNEL.csv',
              'U_test_NlargerP_UNIPALS.csv',
              'T_KERNEL.csv',
              'A_NIPALS.csv',
              'V_NlargerP_KERNEL.csv',
              'P2_UNIPALS.csv',
              'Y_predict_test_SIMPLS.csv',
              'Ts_test_NIPALS.csv',
              'Ts_SIMPLS.csv',
              'beta_NlargerP_KERNEL.csv',
              'A_NlargerP_KERNEL.csv',
              'V_SIMPLS.csv',
              'T_test_KERNEL.csv',
              'U_test_SIMPLS.csv',
              'P1_NIPALS.csv',
              'Ts_test_NlargerP_KERNEL.csv',
              'Ts_KERNEL.csv',
              'Y_predict_test_KERNEL.csv',
              'U_test_KERNEL.csv',
              'beta_NIPALS.csv',
              'V_KERNEL.csv',
              'Y_predict_test_UNIPALS.csv',
              'P1_UNIPALS.csv',
              'P1_NlargerP_UNIPALS.csv',
              'Y_predict_test_NlargerP_KERNEL.csv',
              'P2_KERNEL.csv',
              'U_NIPALS.csv',
              'beta_UNIPALS.csv',
              'Ts_NlargerP_KERNEL.csv',
              'V_NlargerP_UNIPALS.csv',
              'U_UNIPALS.csv',
              'U_test_NlargerP_KERNEL.csv'
              ]

GITHUB_TESTDIR = 'https://github.com/DTUComputeStatisticsAndDataAnalysis/MBPLS/raw/master/mbpls/tests'

def data_path():
    path = os.path.dirname(os.path.abspath(__file__))
    return path

def check_test_files():
    dir = 'test_data'
    file_path = os.path.join(data_path(), dir)
    path_checker(file_path)
    for file in TEST_FILES:
        abs_file_path = os.path.join(file_path, file)
        if file_checker(file, abs_file_path, GITHUB_TESTDIR, dir) == 0:
            pass
        else:
            print('File {0} was not available and could not be downloaded'.format(file))
            return 1
    return 0
