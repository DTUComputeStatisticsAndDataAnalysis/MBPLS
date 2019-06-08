"""
Multi-Block Partial Least Squares (MB-PLS) for Python
=====================================================

The mbpls package contains three multi-block capable algorithms, i.e. KERNEL, NIPALS and UNIPALS, as
well as SIMPLS for fast predictions.

The aim of the package is to provide a unified interface and easy access to these algorithms.
"""

#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# author: Laurent Vermue
# author_email: lauve@dtu.dk
#
#
# License: 3-clause BSD

from . import mbpls, data

__all__ = ["mbpls", "data"]

__version__ = "1.0.2"