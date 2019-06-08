#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

from setuptools import setup, find_packages

import mbpls

NAME = "mbpls"
VERSION = mbpls.__version__
DESCRIPTION = "An implementation of the most common partial least squares algorithms as multi-block methods"
URL = 'https://github.com/DTUComputeStatisticsAndDataAnalysis/MBPLS'
AUTHORS = "Andreas Baum, Laurent Vermue"
AUTHOR_MAILS = "<andba@dtu.dk>, <lauve@dtu.dk>"
LICENSE = 'new BSD'

# This is the lowest tested version. Below might work as well
NUMPY_MIN_VERSION = '1.13.3'
SCIPY_MIN_VERSION = '1.0.0'
SCIKIT_LEARN_MIN_VERSION = '0.21.2'
PANDAS_MIN_VERSION = '0.20.0'

def setup_package():
    with open('README.rst') as f:
        LONG_DESCRIPTION = f.read()
        LONG_DESCRIPTION_CONTENT_TYPE = 'text/x-rst'

    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
          url=URL,
          author=AUTHORS,
          author_email=AUTHOR_MAILS,
          packages = find_packages(),
          license=LICENSE,
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS',
                       'Programming Language :: Python :: 3.5',
                       'Programming Language :: Python :: 3.6',
                       'Programming Language :: Python :: 3.7',
                       # 'Development Status :: 4 - Beta',
                       'Development Status :: 5 - Production/Stable'
                       ],
          install_requires=[
              'numpy>={0}'.format(NUMPY_MIN_VERSION),
              'scipy>={0}'.format(SCIPY_MIN_VERSION),
              'scikit-learn>={0}'.format(SCIKIT_LEARN_MIN_VERSION),
              'pandas>={0}'.format(PANDAS_MIN_VERSION)
                ],
          extras_require={
              'tests': [
                  'pytest'],
              'docs': [
                  'sphinx >= 1.6',
                  'sphinx_rtd_theme',
                  'nbsphinx',
                  'nbsphinx_link'
                    ],
              'extras': [
                  'matplotlib',
              ],
          },
          python_requires='>=3.5',
          )

if __name__ == '__main__':
    setup_package()
