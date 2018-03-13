#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# author: Laurent Vermue
# author_email: lauve@dtu.dk
#
#
# License: 3-clause BSD

from setuptools import setup, find_packages

from pls_package import all_pls

NAME = "universal-pls"
DESCRIPTION = "A set of several partial least squares algorithms"
VERSION = all_pls.__version__
AUTHORS = "Andreas Baum, Laurent Vermue"
AUTHOR_MAILS = "<andba@dtu.dk>, <lauve@dtu.dk>"
LICENSE = 'new BSD'

# This is the lowest tested version. Below might work as well
NUMPY_MIN_VERSION = '1.13.3'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
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
                   'Programming Language :: Python :: 3.6',
                   'Development Status :: 2 - Pre-Alpha'
                   ],
      install_requires=[
          'numpy>={0}'.format(NUMPY_MIN_VERSION)
            ]
      )
