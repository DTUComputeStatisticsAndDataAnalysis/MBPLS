#!/usr/bin/env python
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
DESCRIPTION = "An implementation of the most common partial least squares algorithm as multi-block methods"
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
    LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown',
URL = 'https://github.com/b0nsaii/MBPLS'
AUTHORS = "Andreas Baum, Laurent Vermue"
AUTHOR_MAILS = "<andba@dtu.dk>, <lauve@dtu.dk>"
LICENSE = 'new BSD'

# This is the lowest tested version. Below might work as well
NUMPY_MIN_VERSION = '1.13.3'
MATPLOTLIB_MIN_VERSION = '2.1.1'
SCIPY_MIN_VERSION = '1.0.0'
SCIKIT_LEARN_MIN_VERSION = '0.20.0'

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
                   'Development Status :: 3 - Alpha'
                   ],
      install_requires=[
          'numpy>={0}'.format(NUMPY_MIN_VERSION),
          'matplotlib>={0}'.format(MATPLOTLIB_MIN_VERSION),
          'scipy>={0}'.format(SCIPY_MIN_VERSION),
          'scikit-learn>={0}'.format(SCIKIT_LEARN_MIN_VERSION)
            ],
      python_requires='>=3.5',
      )
