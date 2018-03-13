#!/usr/bin/env python
#
# -*- coding: utf-8 -*-
#
# author: Laurent Vermue
# author_email: lauve@dtu.dk
#
#
# License: 3-clause BSD

""" Base classes for all functions."""

from abc import ABCMeta, abstractmethod
from six import with_metaclass


class BaseEstimator(with_metaclass(ABCMeta), object):
    """Base class for estimators."""

    @abstractmethod
    def fit(self):
        """Method to reset the estimator"""
        raise NotImplementedError("This method has to be defined for the specific class.")

    @abstractmethod
    def transform(self):
        """Method to reset the estimator"""
        raise NotImplementedError("This method has to be defined for the specific class.")

    @abstractmethod
    def predict(self):
        """Method to reset the estimator"""
        raise NotImplementedError("This method has to be defined for the specific class.")


class FitTransform(object):
    def fit_transform(self, X, Y=None, **fit_params):
        """Fit to data, then transform it.
        """
        if Y is None:
            # fit and transform of x only
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit and transform of x and y
            return self.fit(X, Y, **fit_params).transform(X, Y)

