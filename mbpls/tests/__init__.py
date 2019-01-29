"""
The :mod:`mbpls.tests` module contains an installation test script, which contains a range of tests created for the
pytest package that ensure the correct installation of the mbpls package and subsequently the recreation of predefined
results for all methods. This is especially designed to verify the validity of results after making changes to the
source code of the implemented algorithms.
"""

#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

from . import test_mbpls

__all__ = ["test_mbpls.py"]