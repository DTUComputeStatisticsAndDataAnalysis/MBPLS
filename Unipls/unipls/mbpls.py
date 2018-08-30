#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:31:48 2018

@authors: Andreas Baum, andba@dtu.dk; Laurent Vermue, lauve@dtu.dk

"""

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_consistent_length
# Add this class to estimator checks
from sklearn.utils import estimator_checks
estimator_checks.CROSS_DECOMPOSITION.append('MBPLS')
from sklearn import metrics
import collections
import numpy as np
from six import with_metaclass
from sklearn.preprocessing import StandardScaler
from abc import ABCMeta, abstractmethod
from sklearn.exceptions import DataConversionWarning

__all__ = ['MBPLS']


#class PLSRegression(BaseEstimator, TransformerMixin, RegressorMixin):
class MBPLS(BaseEstimator, TransformerMixin, RegressorMixin):
    # TODO: Write a bit about the models used in this text field
    """Multiblock PLS regression

        - super scores are normalized to length 1 (not done in R package ade4)
        - N > P run SVD(X'YY'X) --> PxP matrix
        - N < P run SVD(XX'YY') --> NxN matrix (Lindgreen et al. 1998)

        Model settings
        ----------

        method : string (default 'SVD')
        The method being used to derive the model attributes, possible are 'SVD', 'NIPALS', 'SIMPLS'

        n_components : int
        Number of Latent Variables.

        standardize : bool (default True)
        Standardizing the data

        full_svd : bool (default True)
        Using full singular value decomposition when performing SVD method

        max_tol : non-negative float (default 1e-14)
        Maximum tolerance allowed when using the iterative NIPALS algorithm

        Model attributes after fitting
        ----------
        X side:
        Ts_ - super scores
        T_ - list of block scores (number of elements = number of blocks)
        W_ - list of block loadings (number of elements = number of blocks)
        A_ - super weights/loadings (number of rows = number of blocks)
        eigenv - normal PLS weights (of length 1)
        weights - concatenated eigenv's
        P_ - Loadings

        Y side:
        U_ - scores on Y
        V_ - loadings on Y y_loading_

    """

    def __init__(self, n_components=2, full_svd=True, method='NIPALS', standardize=True, max_tol=1e-14, calc_all=True):
        self.n_components = n_components
        self.full_svd = full_svd
        self.method = method
        self.standardize = standardize
        self.max_tol = max_tol
        self.calc_all = calc_all

    # IDEA: Equal vectorlength for all blocks with enforcement,
    # IDEA: Amount of features corrected block length (Blockimportance corrected) (Andreas)
    # IDEA: Variance related to all Blocks (Andreas)

    def fit(self, X, Y):
        """ Fit model to given data

        Parameters
        ----------

        X : list
            of all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
        Y : array
            1-dim or 2-dim array of reference values
        """

        global U_, T_, R_
        Y = check_array(Y, dtype=np.float64, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if self.standardize:
            self.x_scalers_ = []
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    self.x_scalers_.append(StandardScaler(with_mean=True, with_std=True))
                    # Check dimensions
                    check_consistent_length(X[block], Y)
                    X[block] = check_array(X[block], dtype=np.float64, copy=True)
                    X[block] = self.x_scalers_[block].fit_transform(X[block])
            else:
                self.x_scalers_.append(StandardScaler(with_mean=True, with_std=True))
                # Check dimensions
                X = check_array(X, dtype=np.float64, copy=True)
                check_consistent_length(X, Y)
                X = [self.x_scalers_[0].fit_transform(X)]

            self.y_scaler_ = StandardScaler(with_mean=True, with_std=True)
            Y = self.y_scaler_.fit_transform(Y)

        self.num_blocks_ = len(X)

        # Store start/end feature indices for all x blocks
        feature_indices = []
        for block in X:
            if len(feature_indices) > 0:
                feature_indices.append(np.array([feature_indices[-1][1], feature_indices[-1][1] + block.shape[1]]))
            else:
                feature_indices.append(np.array([0, block.shape[1]]))

        self.W_ = []
        self.W_non_normal_ = []
        self.T_ = []
        self.A_ = np.empty((self.num_blocks_, 0))
        # TODO: A correct
        self.A_corrected_ = np.empty((self.num_blocks_, 0))
        self.explained_var_xblocks_ = np.empty((self.num_blocks_, 0))
        self.V_ = np.empty((Y.shape[1], 0))
        self.loading_y_ = np.empty((Y.shape[1], 0))
        self.U_ = np.empty((Y.shape[0], 0))
        self.Ts_ = np.empty((Y.shape[0], 0))
        self.explained_var_y_ = []
        self.explained_var_x_ = []

        for block in range(self.num_blocks_):
            self.W_.append(np.empty((X[block].shape[1], 0)))
            self.W_non_normal_.append(np.empty((X[block].shape[1], 0)))
            self.T_.append(np.empty((X[block].shape[0], 0)))

        # Concatenate X blocks
        X = np.hstack(X)
        self.P_ = np.empty((X.shape[1], 0))
        self.W_concat_ = np.empty((X.shape[1], 0))
        weights = np.empty((X.shape[1], 0))

        if self.method == 'UNIPALS':
            num_samples = X.shape[0]
            num_features = X.shape[1]

            if num_samples >= num_features:
                for comp in range(self.n_components):
                    # 1. Restore X blocks (for each deflation step)
                    Xblocks = []
                    for indices in feature_indices:
                        Xblocks.append(X[:, indices[0]:indices[1]])

                    # 2. Calculate eigenv (normal pls weights) by SVD(X'YY'X) --> eigenvector with largest eigenvalue
                    S = X.T.dot(Y).dot(Y.T).dot(X)
                    # IDEA: Norming of vectors
                    eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]

                    # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                    w = []
                    a = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        partialloading = eigenv[indices[0]:indices[1]]
                        w.append(partialloading / np.linalg.norm(partialloading))
                        a.append(np.linalg.norm(partialloading) ** 2)

                    # 4. Calculate block scores t1, t2, ... as tn = Xn*wn
                    t = []
                    for block, blockloading in zip(Xblocks, w):
                        t.append(np.dot(block, blockloading))

                    # 5. Calculate super scores ts
                    ts = np.dot(X, eigenv)
                    # IDEA: Concat w
                    ts = ts / np.linalg.norm(ts)

                    # 6. Calculate v (Y-loading) by projection of ts on Y
                    v = np.dot(Y.T, ts) / np.dot(ts.T, ts)

                    # 7. Calculate u (Y-scores)
                    u = np.dot(Y, v)
                    u = u / np.linalg.norm(u)

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = np.dot(X.T, ts) / np.dot(ts.T, ts)
                    varx_explained = (np.dot(ts, p.T) ** 2).sum()
                    if comp == 0:
                        varx = (X ** 2).sum()
                    self.explained_var_x_.append(varx_explained / varx)

                    varx_blocks_explained = []
                    if comp == 0:
                        varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (np.dot(ts, p[indices[0]:indices[1]].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])

                    X = X - np.dot(ts, p.T)

                    # 9. Calculate explained variance in Y
                    vary_explained = (np.dot(ts, v.T) ** 2).sum()
                    vary = (Y ** 2).sum()
                    self.explained_var_y_.append(vary_explained / vary)

                    # 10. Upweight Block Importances of blocks with less features
                    sum_vars = []
                    for vector in w:
                        sum_vars.append(len(vector))
                    if len(a)==1:
                        a_corrected = [1]
                    else:
                        a_corrected = []
                        for bip, sum_var in zip(a, sum_vars):
                            factor = 1 - sum_var / np.sum(sum_vars)
                            a_corrected.append(bip * factor)
                        a_corrected = list(a_corrected / np.sum(a_corrected))

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T_, W_, U_, V_, Ts_, weights, P_ and A_
                    self.V_ = np.hstack((self.V_, v))
                    self.U_ = np.hstack((self.U_, u))
                    self.A_ = np.hstack((self.A_, np.matrix(a).T))
                    self.A_corrected_ = np.hstack((self.A_corrected_, np.matrix(a_corrected).T))
                    self.explained_var_xblocks_ = np.hstack(
                        (self.explained_var_xblocks_, np.matrix(varx_blocks_explained).T))
                    self.Ts_ = np.hstack((self.Ts_, ts))
                    self.P_ = np.hstack((self.P_, p))
                    weights = np.hstack((weights, eigenv))
                    for block in range(self.num_blocks_):
                        self.W_[block] = np.hstack((self.W_[block], w[block]))
                        self.T_[block] = np.hstack((self.T_[block], t[block]))
                    pseudoinv = np.dot(weights, np.linalg.pinv(np.dot(self.P_.T, weights)))
                    self.R_ = pseudoinv
                    pseudoinv = np.dot(pseudoinv, np.linalg.pinv(np.dot(self.Ts_.T, self.Ts_)))
                    pseudoinv = np.dot(pseudoinv, self.Ts_.T)
                    self.beta_ = np.dot(pseudoinv, Y)

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
                    v = np.dot(Y.T, ts) / np.dot(ts.T, ts)

                    # 4. Calculate u (Y-scores)
                    u = np.dot(Y, v)
                    u = u / np.linalg.norm(u)

                    # 5. Calculate weights eigenv
                    eigenv = np.dot(X.T, u)
                    eigenv = eigenv / np.linalg.norm(eigenv)

                    # 6. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                    w = []
                    a = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        partialloading = eigenv[indices[0]:indices[1]]
                        w.append(partialloading / np.linalg.norm(partialloading))
                        a.append(np.linalg.norm(partialloading) ** 2)

                    # 7. Calculate block scores t1, t2, ... as tn = Xn*wn
                    t = []
                    for block, blockloading in zip(Xblocks, w):
                        t.append(np.dot(block, blockloading))

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = np.dot(X.T, ts) / np.dot(ts.T, ts)
                    varx_explained = (np.dot(ts, p.T) ** 2).sum()
                    if comp == 0:
                        varx = (X ** 2).sum()
                    self.explained_var_x_.append(varx_explained / varx)

                    varx_blocks_explained = []
                    if comp == 0:
                        varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (np.dot(ts, p[indices[0]:indices[1]].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])

                    X = X - np.dot(ts, p.T)

                    # 9. Calculate explained variance in Y
                    vary_explained = (np.dot(ts, v.T) ** 2).sum()
                    vary = (Y ** 2).sum()
                    self.explained_var_y_.append(vary_explained / vary)

                    # 10. Upweight Block Importances of blocks with less features (provided as additional figure of merit)
                    sum_vars = []
                    for vector in w:
                        sum_vars.append(len(vector))
                    if len(a)==1:
                        a_corrected = [1]
                    else:
                        a_corrected = []
                        for bip, sum_var in zip(a, sum_vars):
                            factor = 1 - sum_var / np.sum(sum_vars)
                            a_corrected.append(bip * factor)
                        a_corrected = list(a_corrected / np.sum(a_corrected))

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T_, W_, U_, V_, Ts_, weights, P_ and A_
                    self.V_ = np.hstack((self.V_, v))
                    self.U_ = np.hstack((self.U_, u))
                    self.A_ = np.hstack((self.A_, np.matrix(a).T))
                    self.A_corrected_ = np.hstack((self.A_corrected_, np.matrix(a_corrected).T))
                    self.explained_var_xblocks_ = np.hstack(
                        (self.explained_var_xblocks_, np.matrix(varx_blocks_explained).T))
                    self.Ts_ = np.hstack((self.Ts_, ts))
                    self.P_ = np.hstack((self.P_, p))
                    weights = np.hstack((weights, eigenv))
                    for block in range(self.num_blocks_):
                        self.W_[block] = np.hstack((self.W_[block], w[block]))
                        self.T_[block] = np.hstack((self.T_[block], t[block]))
                    pseudoinv = np.dot(weights, np.linalg.pinv(np.dot(self.P_.T, weights)))
                    self.R_ = pseudoinv
                    pseudoinv = np.dot(pseudoinv, np.linalg.pinv(np.dot(self.Ts_.T, self.Ts_)))
                    pseudoinv = np.dot(pseudoinv, self.Ts_.T)
                    self.beta_ = np.dot(pseudoinv, Y)

            return self

        elif self.method == 'KERNEL':
            num_samples = X.shape[0]
            num_features = X.shape[1]

            if num_samples >= num_features:
                # Calculate kernel matrix once
                S = X.T.dot(Y).dot(Y.T).dot(X)
                # Calculate variance covariance matrixes
                VAR = X.T.dot(X)
                COVAR = X.T.dot(Y)
                for comp in range(self.n_components):

                    eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]

                    # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                    w = []
                    a = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        partialloading = eigenv[indices[0]:indices[1]]
                        w.append(partialloading / np.linalg.norm(partialloading))
                        a.append(np.linalg.norm(partialloading) ** 2)

                    # 6. Calculate v (Y-loading) by projection of ts on Y
                    v = eigenv.T.dot(COVAR)/eigenv.T.dot(VAR).dot(eigenv)

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = eigenv.T.dot(VAR)/eigenv.T.dot(VAR).dot(eigenv)

                    # Update kernel and variance/covariance matrices
                    deflate_matrix = np.eye(S.shape[0])-eigenv.dot(p)
                    S = deflate_matrix.T.dot(S).dot(deflate_matrix)
                    VAR = deflate_matrix.T.dot(VAR).dot(deflate_matrix)
                    # TODO: Deflation of y variables is not necessary
                    COVAR = deflate_matrix.T.dot(COVAR)

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T_, W_, U_, V_, Ts_, weights, P_ and A_
                    self.V_ = np.hstack((self.V_, v.T))
                    self.A_ = np.hstack((self.A_, np.matrix(a).T))
                    self.P_ = np.hstack((self.P_, p.T))
                    self.W_concat_ = np.hstack((self.W_concat_, eigenv))
                    for block in range(self.num_blocks_):
                        self.W_[block] = np.hstack((self.W_[block], w[block]))

                # TODO: The authors actually implemented a more efficient algorithm without inversion
                self.R_ = self.W_concat_.dot(np.linalg.pinv(self.P_.T.dot(self.W_concat_)))
                self.beta_ = self.R_.dot(self.V_.T)

            if num_features > num_samples:
                # Calculate association matrices
                AS_X = X.dot(X.T)
                AS_Y = Y.dot(Y.T)
                # Calculate kernel matrix once
                S = AS_X.dot(AS_Y)

                for comp in range(self.n_components):

                    # Calculate the eigenvector with the largest eigenvalue of the kernel matrix
                    ts = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]

                    # Calculate response-score vector
                    # IDEA: Leave out y deflation
                    u = AS_Y.dot(ts)
                    u = u / np.linalg.norm(u)

                    # Deflate association and kernel matrices
                    deflate_matrix = np.eye(S.shape[0], S.shape[0])-ts.dot(ts.T)
                    AS_X = deflate_matrix.dot(AS_X).dot(deflate_matrix)
                    # IDEA: Leave out y deflation
                    AS_Y = deflate_matrix.dot(AS_Y).dot(deflate_matrix)
                    S = AS_X.dot(AS_Y)

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T_, W_, U_, V_, Ts_, weights, P_ and A_
                    self.U_ = np.hstack((self.U_, u))
                    self.Ts_ = np.hstack((self.Ts_, ts))

                self.W_concat_ = X.T.dot(self.U_)
                # Normalize weights to length one per column
                self.W_concat_ = self.W_concat_ / np.linalg.norm(self.W_concat_, axis=0)
                self.P_ = (X.T.dot(self.Ts_)).dot(np.linalg.pinv(self.Ts_.T.dot(self.Ts_)))
                self.V_ = (Y.T.dot(self.Ts_)).dot(np.linalg.pinv(self.Ts_.T.dot(self.Ts_)))

                self.R_ = self.W_concat_.dot(np.linalg.pinv(self.P_.T.dot(self.W_concat_)))
                self.beta_ = self.R_.dot(self.V_.T)

                if self.calc_all:
                    # Calculate Block importances
                    for component in range(self.n_components):
                        a = []
                        w = []
                        t = []
                        for indices in feature_indices:
                            partialloading = self.W_concat_[indices[0]:indices[1], component:component + 1]
                            weight = partialloading / np.linalg.norm(partialloading)
                            w.append(weight)
                            a.append(np.linalg.norm(partialloading) ** 2)
                            # Calculate block scores t1, t2, ... as tn = Xn*wn
                            t.append(X[:, indices[0]:indices[1]].dot(weight))

                        # TODO: Check if this could be done more efficiently
                        # Deflate X matrix to allow correct score calculation
                        X = X - self.Ts_[:, component:component + 1].dot(self.P_[:, component:component+1].T)
                        self.A_ = np.hstack((self.A_, np.matrix(a).T))
                        for block in range(self.num_blocks_):
                            self.W_[block] = np.hstack((self.W_[block], w[block]))
                            self.T_[block] = np.hstack((self.T_[block], t[block]))

            return self

        elif self.method == 'NIPALS':

            # Restore X blocks (for each deflation step)
            Xblocks = []
            for indices in feature_indices:
                Xblocks.append(X[:, indices[0]:indices[1]])
            Y_calc = Y

            # Wangen and Kowalski (1988)
            for comp in range(self.n_components):
                # 0. Take first column vector out of y and regress against each block
                u_a = Y_calc[:, 0:1]
                run = 1
                diff_t = 1
                while diff_t > self.max_tol:  # Condition on error of ts
                    # 1. Regress u_a against all blocks
                    weights = []
                    weights_non_normal = []
                    for block in range(self.num_blocks_):
                        weights.append(np.dot(Xblocks[block].T, u_a) / np.dot(u_a.T, u_a))
                        weights_non_normal.append(np.dot(Xblocks[block].T, u_a) / np.dot(u_a.T, u_a))
                        # normalize block weigths
                        weights[block] = weights[block] / np.linalg.norm(weights[block])
                    # 2. Regress block weights against rows of each block
                    scores = []
                    for block in range(self.num_blocks_):
                        # Diverging from Wangen and Kowalski by using regression instead of dividing by number of components
                        scores.append(np.dot(Xblocks[block], weights[block]) / np.dot(weights[block].T, weights[block]))
                    # 3. Append all block scores in T_
                    T_ = np.hstack((scores))
                    # 4. Regress u_a against block of block scores
                    superweights = np.dot(T_.T, u_a) / np.dot(u_a.T, u_a)
                    superweights = superweights / np.linalg.norm(superweights)
                    # 5. Regress superweights against T_ to obtain superscores
                    superscores = np.dot(T_, superweights) / np.dot(superweights.T, superweights)
                    superscores = superscores / np.linalg.norm(superscores)
                    if run == 1:
                        pass
                    else:
                        diff_t = np.sum(superscores_old - superscores)
                    superscores_old = np.copy(superscores)
                    # 6. Regress superscores agains Y_calc
                    response_weights = np.dot(Y_calc.T, superscores) / np.dot(superscores.T, superscores)
                    # 7. Regress response_weights against Y
                    response_scores = np.dot(Y_calc, response_weights) / np.dot(response_weights.T, response_weights)
                    response_scores = response_scores / np.linalg.norm(response_scores)
                    u_a = response_scores
                    run += 1

                # 8. Calculate loading
                loadings = [None] * self.num_blocks_
                for block in range(self.num_blocks_):
                    loadings[block] = np.dot(Xblocks[block].T, superscores) / np.dot(superscores.T, superscores)
                loading_y_ = Y_calc.T.dot(response_scores) / response_scores.T.dot(response_scores)
                # 9. Deflate X_calc
                for block in range(self.num_blocks_):
                    Xblocks[block] = Xblocks[block] - np.dot(superscores, loadings[block].T)

                # 10. Deflate Y - No deflation of Y
                #Y_calc = Y_calc - np.dot(superscores, response_weights.T)

                # 11. Append the resulting vectors
                self.V_ = np.hstack((self.V_, response_weights))
                self.loading_y_ = np.hstack((self.loading_y_, loading_y_))
                self.U_ = np.hstack((self.U_, response_scores))
                # IDEA: Shouldn't this be the quotient
                # superweights = superweights / np.sum(superweights, axis=0)
                # self.A_ = np.hstack((self.A_, superweights / np.sum(superweights, axis=0)))
                self.A_ = np.hstack((self.A_, superweights ** 2))  # squared for total length 1
                self.Ts_ = np.hstack((self.Ts_, superscores))
                # Concatenate the block-loadings
                loadings = np.vstack(loadings)
                self.P_ = np.hstack((self.P_, loadings))
                for block in range(self.num_blocks_):
                    self.W_[block] = np.hstack((self.W_[block], weights[block] * -1))
                    self.W_non_normal_[block] = np.hstack((self.W_non_normal_[block], weights_non_normal[block] * -1))
                    self.T_[block] = np.hstack((self.T_[block], scores[block] * -1))

            # Negate results to achieve same results as with SVD
            # TODO: Check if this makes sense
            # self.Ts_ *= -1
            # self.P_ *= -1
            # self.U_ *= -1
            # self.V_ *= -1
            # self.loading_y_ *= -1

            weights_total = np.concatenate((self.W_non_normal_), axis=0)  # Concatenate weights for beta_ calculation
            weights_total = weights_total / np.linalg.norm(weights_total, axis=0)
            pseudoinv = np.linalg.pinv((np.dot(self.P_.T, weights_total)))
            R_ = np.dot(weights_total, pseudoinv)
            self.R_ = R_
            self.beta_ = np.dot(R_, self.V_.T)

            # Testing R_y
            # weights_y_norm = self.V_ / np.linalg.norm(self.V_, axis=0)
            # pseudoinv_y = np.linalg.pinv((np.dot(self.loading_y_.T, weights_y_norm)))
            # R_y = np.dot(weights_y_norm, pseudoinv_y)
            # self.R_y = R_y

            return self

        elif self.method == 'SIMPLS':
            # de Jong 1993
            S = np.dot(X.T, Y)
            for comp in range(self.n_components):
                q = np.linalg.svd(S.T.dot(S), full_matrices=self.full_svd)[0][:, 0:1]
                r = S.dot(q)
                w = r.copy()
                t = X.dot(r)
                t = t - np.mean(t)
                normt = np.sqrt(t.T.dot(t))
                t = t / normt
                r = r / normt
                p = X.T.dot(t)
                q = Y.T.dot(t)
                u = Y.dot(q)
                v = p
                if comp > 0:
                    v = v - V_.dot(V_.T.dot(p))
                    u = u - T_.dot(T_.T.dot(u))
                v = v / np.sqrt(v.T.dot(v))
                S = S - v.dot(v.T.dot(S))
                
                if comp == 0:
                    R_ = r
                    T_ = t
                    P_ = p
                    Q = q
                    U_ = u / np.linalg.norm(u)
                    V_ = v
                    W = w
                else:
                    R_ = np.hstack((R_, r))
                    T_ = np.hstack((T_, t))
                    P_ = np.hstack((P_, p))
                    Q = np.hstack((Q, q))
                    U_ = np.hstack((U_, u / np.linalg.norm(u)))
                    V_ = np.hstack((V_, v))
                    W = np.hstack((W, w))
                    
            self.P_ = P_
            self.Ts_ = T_
            self.U_ = U_
            self.R_ = R_
            self.beta_ = R_.dot(Q.T)
            self.V_ = Q
            self.W_ = W

            return self
        else:
            raise NameError('Method you called is unknown')

    def transform(self, X, Y=None):
        """ Obtain scores based on the fitted model

         Parameters
        ----------
        X : list
            of all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
        (optional) Y : array
            1-dim or 2-dim array of reference values

        Returns
        ----------
        X_scores : list
        List of np.arrays for several blocks

        Y_scores : np.array (optional)
        Y-scores, if y was given
        """
        check_is_fitted(self, 'beta_')

        #assert isinstance(X, list), "The different blocks have to be passed in a list"
        # TODO: Return scores per Block

        if self.standardize:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64)
                    X[block] = self.x_scalers_[block].transform(X[block])
            else:
                # Check dimensions
                X = check_array(X, dtype=np.float64)
                X = [self.x_scalers_[0].transform(X)]

            X = np.hstack(X)

            if Y is not None:
                Y = check_array(Y, dtype=np.float64, ensure_2d=False)
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                Y = self.y_scaler_.transform(Y)
                return X.dot(self.R_), Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
            else:
                return X.dot(self.R_)

        else:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64)
            else:
                # Check dimensions
                X = [check_array(X, dtype=np.float64)]

            X = np.hstack(X)

            if Y is not None:
                Y = check_array(Y, dtype=np.float64, ensure_2d=False)
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                return X.dot(self.R_), Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
            else:
                return X.dot(self.R_)

    def predict(self, X):
        """Predict y based on the fitted model

        Parameters
        ----------
        X : list
            of all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables

        Returns
        ----------
        y_hat : np.array
        Predictions made based on trained model and supplied X

        """
        check_is_fitted(self, 'beta_')

        if self.standardize:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64)
                    X[block] = self.x_scalers_[block].transform(X[block])
            else:
                X = check_array(X, dtype=np.float64)
                X = [self.x_scalers_[0].fit_transform(X)]


            X = np.hstack(X)
            y_hat = self.y_scaler_.inverse_transform(X.dot(self.beta_))
        else:
            X = np.hstack(X)
            y_hat = X.dot(self.beta_)

        return y_hat.squeeze()

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and then, then transform the given data to lower dimensions.
        """
        # fit and transform of x and y
        return self.fit(X, y, **fit_params).transform(X, y)

    def fit_predict(self, X, Y, **fit_params):
        """Fit to data, then predict it.
        """
        # TODO: Standardisierte Ergebnisse ausgeben
        # fit the model to x and y and return their scores
        return self.fit(X, Y, **fit_params).predict(X)

    def r2_score(self, X, Y):
        if self.standardize:
            return metrics.r2_score(Y, self.predict(X))
        else:
            # When the data is not standardized, the variables have to be variance weighted
            return metrics.r2_score(Y, self.predict(X), sample_weight=None, multioutput='variance_weighted')

    def explained_variance_score(self, X, Y):
        if self.standardize:
            return metrics.explained_variance_score(Y, self.predict(X))
        else:
            # When the data is not standardized, the variables have to be variance weighted
            return metrics.explained_variance_score(Y, self.predict(X), sample_weight=None, multioutput='variance_weighted')

    def plot(self, num_components='Component that should be plotted'):
        """
        Function that prints the fitted values of the instance.
        INPUT:
        num_components: Int or list
                        Int: The number of components that will be plotted, starting with the first component
                        list: Indices or range of the components that should be plotted
        ----------

        """
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib import ticker
        #plt.close('all')

        # Check arguments and create lists
        if isinstance(num_components, int):
            num_components = np.arange(num_components)
        else:
            try:
                num_components = np.array(num_components)
                num_components = num_components - 1
                # Create python indexing

            except AssertionError as e:
                e.args += ('The num_components argument had the wrong format, please try again',)
                raise

        # Check the length of the requested items
        if len(num_components) > self.n_components:
            print("You requested more components to be plotted than your fitted model has.")
            print("The requested list will be shortened to the maximum amount of components possible")
            num_components = num_components[0:self.n_components]
            if max(num_components) + 1 > self.n_components:
                raise ValueError("You requested not existing indices.")

        # Iterate over all required components
        for comp in num_components:
            fig = plt.figure()
            plt.suptitle("Component {:d}".format(comp + 1), fontsize=14, fontweight='bold')

            gs1 = GridSpec(1, self.num_blocks_, top=0.875, bottom=0.85, right=0.95)
            for block in range(self.num_blocks_):
                plt.subplot(gs1[0, block])
                plt.text(0.5, 0.5, "X-Block {:d}\nImportance: {:.0f}%".format(block + 1, self.A_[block, comp] * 100),
                         fontsize=12, horizontalalignment='center')
                plt.axis('off')

            gs2 = GridSpec(2, self.num_blocks_, top=0.8, hspace=0.45, wspace=0.45, right=0.95)
            loading_axes = []
            score_axes = []
            # List for inverse transforming the loadings/weights
            W_inv_trans = []
            for block in range(self.num_blocks_):
                # Inverse transforming weights/loadings
                # TODO: Does this make sense?
                if self.standardize:
                    W_inv_trans.append(self.x_scalers_[block].inverse_transform(self.W_[block][:, comp]))
                    #W_inv_trans.append(self.W_[block][:, comp])
                else:
                    W_inv_trans.append(self.W_[block][:, comp])
                #W_inv_trans.append(self.P_[block][:, comp])


                if len(loading_axes) == 0:
                    # Loadings
                    loading_axes.append(plt.subplot(gs2[0, block]))
                    plt.plot(W_inv_trans[block])
                    step = int(W_inv_trans[block].shape[0] / 4)
                    plt.xticks(np.arange(0, W_inv_trans[block].shape[0], step),
                               np.arange(1, W_inv_trans[block].shape[0] + 1, step))
                    # loading_axes[block].yaxis.set_major_formatter(ticker.FormatStrFormatter('%4.2f'))
                    plt.grid()
                    plt.ylabel("Loading")
                    plt.xlabel("Variable")

                    # Scores
                    score_axes.append(plt.subplot(gs2[1, block]))
                    plt.plot(self.T_[block][:, comp])
                    step = int(self.T_[block].shape[0] / 4)
                    plt.xticks(np.arange(0, self.T_[block].shape[0], step),
                               np.arange(1, self.T_[block].shape[0] + 1, step))
                    plt.ylabel("Score")
                    plt.xlabel("Sample")
                    plt.grid()
                else:
                    # Loadings
                    loading_axes.append(plt.subplot(gs2[0, block]))
                    plt.plot(W_inv_trans[block])
                    step = int(W_inv_trans[block].shape[0] / 4)
                    plt.xticks(np.arange(0, W_inv_trans[block].shape[0], step),
                               np.arange(1, W_inv_trans[block].shape[0] + 1, step))
                    # plt.setp(loading_axes[block].get_yticklabels(), visible=False)
                    plt.xlabel("Variable")
                    plt.grid()
                    # Scores
                    score_axes.append(plt.subplot(gs2[1, block]))
                    plt.plot(self.T_[block][:, comp])
                    step = int(self.T_[block].shape[0] / 4)
                    plt.xticks(np.arange(0, self.T_[block].shape[0], step),
                               np.arange(1, self.T_[block].shape[0] + 1, step))
                    # plt.setp(score_axes[block].get_yticklabels(), visible=False)
                    plt.xlabel("Sample")
                    plt.grid()

            plt.show()

        plt.suptitle("Block importances", fontsize=14, fontweight='bold')
        gs3 = GridSpec(1, 1, top=0.825, right=0.7, hspace=0.45, wspace=0.4)
        ax = plt.subplot(gs3[0, 0])
        width = 0.8 / len(num_components)
        for i, comp in enumerate(num_components):
            ax.bar(np.arange(self.num_blocks_) + 0.6 + i * width, 100 * np.ravel(self.A_[:, comp]), width=width, \
                   label="Component {}".format(comp + 1))
        ax.set_xticklabels(list(np.arange(self.num_blocks_) + 1))
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.4))
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_xlabel("Block")
        ax.set_ylabel("Block importance in %")
        plt.show()
