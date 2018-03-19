#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:31:48 2018

data from University of Copenhagen, Frans van den Berg
http://www.models.life.ku.dk/~courses/MBtoolbox/mbtmain.htm

info from data origin:
X (objects x all X-block variables) one, augmented data matrix with all the X-blocks
Xin (number of X-blocks x 2 = first and last variable per block) index for X-blocks
(optional:)
Xpp (number of X-blocks x 3) processing per X-block
 - first entry scaling: 0=do not scale to sum-of-squares, number=scale to sum-of-squares
 - second entry preprocessing: 0=none, 1=mean centering, 2=auto scaling
 - third entry = number of factors to compute for this X-block
Y (objects x all Y-block variables) one, augmented data matrix with all the Y-blocks
Yin (number of Y-blocks x 2 = first and last variable per block) index for Y-blocks
 - if omitted, one Y-block is assumed
Ypp (number of Y-blocks x 1) preprocessing per Y-block: 0=none, 1=mean centering, 2=auto scaling

Model (1 x 2) first entry : 0 = no cross validation (default), 1 = cross validation;
  second entry overrides default modeling methods (MBCPCA for X, MBSPLS for X and Y)

- 0=MBCPCA, 1=MBPCA, 2=MBHPCA

- 10=MBSPLS, 11=MBBPLS, 12=MBHPLS

@author: Andreas Baum, andba@dtu.dk
edited: Laurent Vermue, lauve@dtu.dk

"""

from .base import BaseEstimator, FitTransform
import numpy as np

__all__ = ['MBPLS']


class MBPLS(BaseEstimator, FitTransform):
    """Multiblock PLS regression

        - super scores are normalized to length 1 (not done in R package ade4)
        - N > P run SVD(X'YY'X) --> PxP matrix
        - N < P run SVD(XX'YY') --> NxN matrix (Lindgreen et al. 1998)
        - to do: export regression vector b

        ----------

        X : list
            of all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
        Y : array
            1-dim or 2-dim array of reference values
        n_components : int
            Number of Latent Variables.


        X side:
        Ts - super scores
        T - list of block scores (number of elements = number of blocks)
        W - list of block loadings (number of elements = number of blocks)
        A - super weights/loadings (number of rows = number of blocks)
        eigenv - normal PLS weights (of length 1)
        weights - concatenated eigenv's
        P - Loadings

        Y side:
        U - scores on Y
        V - loadings on Y

    """

    def __init__(self, n_components, full_svd=True):
        self.n_components = n_components
        self.full_svd = full_svd

    def fit(self, X, Y):
        num_blocks = len(X)

        # Store start/end feature indices for all x blocks
        feature_indices = []
        for block in X:
            if len(feature_indices) > 0:
                feature_indices.append(np.array([feature_indices[-1][1], feature_indices[-1][1] + block.shape[1]]))
            else:
                feature_indices.append(np.array([0, block.shape[1]]))

        self.W = []
        self.T = []
        self.A = np.empty((num_blocks, 0))
        self.V = np.empty((Y.shape[1], 0))
        self.U = np.empty((Y.shape[0], 0))
        self.Ts = np.empty((Y.shape[0], 0))

        for block in range(num_blocks):
            self.W.append(np.empty((X[block].shape[1], 0)))
            self.T.append(np.empty((X[block].shape[0], 0)))

        # Concatenate X blocks
        X = np.hstack(X)

        self.P = np.empty((X.shape[1], 0))
        weights = np.empty((X.shape[1], 0))

        num_samples = X.shape[0]
        num_features = X.shape[1]
        if num_samples >= num_features:
            for comp in range(self.n_components):
                # 1. Restore X blocks (for each deflation step)
                Xblocks = []
                for indices in feature_indices:
                    Xblocks.append(X[:, indices[0]:indices[1]])

                # 2. Calculate eigenv (normal pls weights) by SVD(X'YY'X) --> eigenvector with largest eigenvalue
                S = np.dot(np.dot(np.dot(X.T, Y), Y.T), X)
                eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]

                # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                w = []
                a = []
                for indices, block in zip(feature_indices, range(num_blocks)):
                    partialloading = eigenv[indices[0]:indices[1]]
                    w.append(partialloading / np.linalg.norm(partialloading))
                    a.append(np.linalg.norm(partialloading))

                # 4. Calculate block scores t1, t2, ... as tn = Xn*wn
                t = []
                for block, blockloading in zip(Xblocks, w):
                    t.append(np.dot(block, blockloading))

                # 5. Calculate super scores ts
                ts = np.dot(X, eigenv)
                ts = ts / np.linalg.norm(ts)

                # 6. Calculate v (Y-loading) by projection of ts on Y
                v = np.dot(Y.T, ts)
                v = v / np.linalg.norm(v)

                # 7. Calculate u (Y-scores)
                u = np.dot(Y, v)

                # 8. Deflate X by calculating: Xnew = X - ts*loading (you need to find a loading for deflation by projecting the scores onto X)
                loading = np.dot(X.T, ts) / np.sqrt(np.dot(ts.T, ts))
                X = X - np.dot(ts, loading.T)

                # 9. Deflate Y by calculating: Ynew = Y - ts*eigenvy.T (deflation on Y is optional)
                # Y = Y - np.dot(ts, v.T)

                # 10. add t, w, u, v, ts, eigenv, loading and a to T, W, U, V, Ts, weights, P and A
                self.V = np.hstack((self.V, v))
                self.U = np.hstack((self.U, u))
                self.A = np.hstack((self.A, np.matrix(a).T))
                self.Ts = np.hstack((self.Ts, ts))
                self.P = np.hstack((self.P, loading))
                weights = np.hstack((weights, eigenv))
                for block in range(num_blocks):
                    self.W[block] = np.hstack((self.W[block], w[block]))
                    self.T[block] = np.hstack((self.T[block], t[block]))

                pseudoinv = np.dot(weights, np.linalg.pinv(np.dot(self.P.T, weights)))
                pseudoinv = np.dot(pseudoinv, np.linalg.pinv(np.dot(self.Ts.T, self.Ts)))
                pseudoinv = np.dot(pseudoinv, self.Ts.T)
                self.beta = np.dot(pseudoinv, Y)

        if num_features > num_samples:
            for comp in range(self.n_components):
                # 1. Restore X blocks (for each deflation step)
                Xblocks = []
                for indices in feature_indices:
                    Xblocks.append(X[:, indices[0]:indices[1]])

                # 2. Calculate ts by SVD(XX'YY') --> eigenvector with largest eigenvalue
                S = np.dot(np.dot(np.dot(X, X.T), Y), Y.T)
                ts = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]

                # 3. Calculate v (Y-loading) by projection of ts on Y
                v = np.dot(Y.T, ts)
                v = v / np.linalg.norm(v)

                # 4. Calculate u (Y-scores)
                u = np.dot(Y, v)

                # 5. Calculate weights eigenv
                eigenv = np.dot(X.T, u)
                eigenv = eigenv / np.linalg.norm(eigenv)

                # 6. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                w = []
                a = []
                for indices, block in zip(feature_indices, range(num_blocks)):
                    partialloading = eigenv[indices[0]:indices[1]]
                    w.append(partialloading / np.linalg.norm(partialloading))
                    a.append(np.linalg.norm(partialloading))

                # 7. Calculate block scores t1, t2, ... as tn = Xn*wn
                t = []
                for block, blockloading in zip(Xblocks, w):
                    t.append(np.dot(block, blockloading))

                # 8. Deflate X by calculating: Xnew = X - ts*loading (you need to find a loading for deflation by projecting the scores onto X)
                loading = np.dot(X.T, ts) / np.sqrt(np.dot(ts.T, ts))
                X = X - np.dot(ts, loading.T)

                # 9. Deflate Y by calculating: Ynew = Y - ts*eigenvy.T (deflation on Y is optional)
                # Y = Y - np.dot(ts, v.T)

                # 10. add t, w, u, v, ts, eigenv, loading and a to T, W, U, V, Ts, weights, P and A
                self.V = np.hstack((self.V, v))
                self.U = np.hstack((self.U, u))
                self.A = np.hstack((self.A, np.matrix(a).T))
                self.Ts = np.hstack((self.Ts, ts))
                self.P = np.hstack((self.P, loading))
                weights = np.hstack((weights, eigenv))

                for block in range(num_blocks):
                    self.W[block] = np.hstack((self.W[block], w[block]))
                    self.T[block] = np.hstack((self.T[block], t[block]))

                pseudoinv = np.dot(weights, np.linalg.pinv(np.dot(self.P.T, weights)))
                pseudoinv = np.dot(pseudoinv, np.linalg.pinv(np.dot(self.Ts.T, self.Ts)))
                pseudoinv = np.dot(pseudoinv, self.Ts.T)
                self.beta = np.dot(pseudoinv, Y)

        return self

    def transform(self, X, Y):
        return self

    def predict(self, X, Y):
        return self

    def plot(self, component='Component that should be plotted'):
        """
        Function that prints the fitted values of the instance.
        """
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(221)
        plt.plot(self.W[0][:, component-1])
        plt.title('block loading x1\nBlock importance: ' + str(round(self.A[0, component-1] ** 2, 2)))
        plt.subplot(222)
        plt.plot(self.W[1][:, component-1])
        plt.title('block loading x2\nBlock importance: ' + str(round(self.A[1, component-1] ** 2, 2)))
        plt.subplot(223)
        plt.plot(self.T[0][:, component-1])
        plt.title('block scores x1')
        plt.subplot(224)
        plt.plot(self.T[1][:, component-1])
        plt.title('block scores x2')
        plt.show()
