"""
The :mod:`mbpls.data` module contains methods to load data real world datasets and to
create artificial data than can be used to test the mbpls methods.
"""

#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

from .get_data import data_path
from .get_data import orthogonal_data
from .get_data import load_CarbohydrateMicroarrays_Data
from .get_data import load_FTIR_Data
from .get_data import load_Intro_Data

__all__ = ["data_path",
           "orthogonal_data",
           "load_CarbohydrateMicroarrays_Data",
           "load_FTIR_Data",
           "load_Intro_Data"
           ]
