#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:31:48 2018

@authors: Andreas Baum, andba@dtu.dk; Laurent Vermue, lauve@dtu.dk

"""

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_consistent_length
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds

__all__ = ['MBPLS']


class MBPLS(BaseEstimator, TransformerMixin, RegressorMixin):
    # TODO: Write a bit about the models used in help
    """Multiblock PLS regression

        - super scores are normalized to length 1 (not done in R package ade4)
        - N > P run SVD(X'YY'X) --> PxP matrix
        - N < P run SVD(XX'YY') --> NxN matrix (Lindgreen et al. 1998)

        Model settings
        ----------

        method : string (default 'NIPALS')
        The method being used to derive the model attributes, possible are 'UNIPALS', 'NIPALS', 'SIMPLS'

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

    def __init__(self, n_components=2, full_svd=False, method='NIPALS', standardize=True, max_tol=1e-14, calc_all=True):
        self.n_components = n_components
        self.full_svd = full_svd
        self.method = method
        self.standardize = standardize
        self.max_tol = max_tol
        self.calc_all = calc_all

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
        if self.method is not 'SIMPLS':
            self.A_ = np.empty((self.num_blocks_, 0))
            self.A_corrected_ = np.empty((self.num_blocks_, 0))
            self.T_ = []
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
            if self.method is not 'SIMPLS':
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
                    if self.full_svd:
                        eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]
                    else:
                        eigenv = svds(S, k=1)[0]

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
                        t.append(block.dot(blockloading))

                    # 5. Calculate super scores ts
                    ts = X.dot(eigenv)
                    ts = ts / np.linalg.norm(ts)

                    # 6. Calculate v (Y-loading) by projection of ts on Y
                    v = Y.T.dot(ts) / ts.T.dot(ts)

                    # 7. Calculate u (Y-scores)
                    u = Y.dot(v)
                    u = u / np.linalg.norm(u)

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = X.T.dot(ts) / ts.T.dot(ts)

                    varx_explained = (ts.dot(p.T) ** 2).sum()
                    if comp == 0:
                        varx = (X ** 2).sum()
                    self.explained_var_x_.append(varx_explained / varx)

                    varx_blocks_explained = []
                    if comp == 0:
                        varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (ts.dot(p[indices[0]:indices[1]].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])

                    X = X - ts.dot(p.T)

                    # 9. Calculate explained variance in Y
                    vary_explained = (ts.dot(v.T) ** 2).sum()
                    vary = (Y ** 2).sum()
                    self.explained_var_y_.append(vary_explained / vary)

                    # 10. Upweight Block Importances of blocks with less features
                    sum_vars = []
                    for vector in w:
                        sum_vars.append(len(vector))
                    if len(a) == 1:
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
                    self.R_ = weights.dot(np.linalg.pinv(self.P_.T.dot(weights)))
                    self.beta = self.R_.dot(self.V_.T)
                # restore blockwise loadings
                self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]

            if num_features > num_samples:
                for comp in range(self.n_components):
                    # 1. Restore X blocks (for each deflation step)
                    Xblocks = []
                    for indices in feature_indices:
                        Xblocks.append(X[:, indices[0]:indices[1]])

                    # 2. Calculate ts by SVD(XX'YY') --> eigenvector with largest eigenvalue
                    S = X.dot(X.T).dot(Y).dot(Y.T)
                    if self.full_svd:
                        ts = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]
                    else:
                        ts = svds(S, k=1)[0]

                    # 3. Calculate v (Y-loading) by projection of ts on Y
                    v = Y.T.dot(ts) / ts.T.dot(ts)

                    # 4. Calculate u (Y-scores)
                    u = Y.dot(v)
                    u = u / np.linalg.norm(u)

                    # 5. Calculate weights eigenv
                    eigenv = X.T.dot(u)
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
                        t.append(block.dot(blockloading))

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = X.T.dot(ts) / ts.T.dot(ts)
                    varx_explained = (ts.dot(p.T) ** 2).sum()
                    if comp == 0:
                        varx = (X ** 2).sum()
                    self.explained_var_x_.append(varx_explained / varx)

                    varx_blocks_explained = []
                    if comp == 0:
                        varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (ts.dot(p[indices[0]:indices[1]].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])

                    X = X - ts.dot(p.T)

                    # 9. Calculate explained variance in Y
                    vary_explained = (ts.dot(v.T) ** 2).sum()
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

                    self.R_ = weights.dot(np.linalg.pinv(self.P_.T.dot(weights)))
                    self.beta_ = self.R_.dot(self.V_.T)
                # restore blockwise loadings
                self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]

            return self

        elif self.method == 'KERNEL':
            num_samples = X.shape[0]
            num_features = X.shape[1]

            if num_samples >= num_features:
                # Based on [1] F. Lindgren, P. Geladi, and S. Wold, “The kernel algorithm for PLS,” J. Chemom.,
                # vol. 7, no. 1, pp. 45–59, Jan. 1993.
                # Calculate kernel matrix once
                S = X.T.dot(Y).dot(Y.T).dot(X)
                # Calculate variance covariance matrixes
                VAR = X.T.dot(X)
                COVAR = X.T.dot(Y)
                for comp in range(self.n_components):
                    if self.full_svd:
                        eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]
                    else:
                        eigenv = svds(S, k=1)[0]

                    # 6. Calculate v (Y-loading) by projection of ts on Y
                    v = eigenv.T.dot(COVAR) / eigenv.T.dot(VAR).dot(eigenv)

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = eigenv.T.dot(VAR) / eigenv.T.dot(VAR).dot(eigenv)

                    if self.calc_all:
                        # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                        w = []
                        a = []
                        for indices, block in zip(feature_indices, range(self.num_blocks_)):
                            partialloading = eigenv[indices[0]:indices[1]]
                            w.append(partialloading / np.linalg.norm(partialloading))
                            self.W_[block] = np.hstack((self.W_[block], w[block]))
                            a.append(np.linalg.norm(partialloading) ** 2)

                        # 10. Upweight Block Importances of blocks with less features
                        sum_vars = []
                        for vector in w:
                            sum_vars.append(len(vector))
                        if len(a) == 1:
                            a_corrected = [1]
                        else:
                            a_corrected = []
                            for bip, sum_var in zip(a, sum_vars):
                                factor = 1 - sum_var / np.sum(sum_vars)
                                a_corrected.append(bip * factor)
                            a_corrected = list(a_corrected / np.sum(a_corrected))
                        self.A_corrected_ = np.hstack((self.A_corrected_, np.matrix(a_corrected).T))
                        u = Y.dot(v.T) / v.dot(v.T)
                        u = u / np.linalg.norm(u)
                        self.U_ = np.hstack((self.U_, u))
                        self.A_ = np.hstack((self.A_, np.matrix(a).T))

                    # Update kernel and variance/covariance matrices
                    deflate_matrix = np.eye(S.shape[0]) - eigenv.dot(p)
                    S = deflate_matrix.T.dot(S).dot(deflate_matrix)
                    VAR = deflate_matrix.T.dot(VAR).dot(deflate_matrix)
                    COVAR = deflate_matrix.T.dot(COVAR)


                    # 11. add t, w, u, v, ts, eigenv, loading and a to T_, W_, U_, V_, Ts_, weights, P_ and A_
                    self.V_ = np.hstack((self.V_, v.T))
                    self.P_ = np.hstack((self.P_, p.T))
                    self.W_concat_ = np.hstack((self.W_concat_, eigenv))

                # IDEA: The authors actually implemented a more efficient algorithm without inversion
                self.R_ = self.W_concat_.dot(np.linalg.pinv(self.P_.T.dot(self.W_concat_)))
                self.beta_ = self.R_.dot(self.V_.T)
                self.Ts_ = X.dot(self.R_)
                # TODO: This needs explanation in the paper (Ts-Norming deviating from the original paper)
                Ts_norm = np.linalg.norm(self.Ts_, axis=0)
                self.V_ = self.V_ * Ts_norm
                self.P_ = self.P_ * Ts_norm
                ## End
                self.Ts_ = self.Ts_ / Ts_norm
                # Calculate extra variables
                # IDEA: Check if this could be done more efficiently
                if self.calc_all:
                    for component in range(self.n_components):
                        if component == 0:
                            varx = (X ** 2).sum()
                            vary = (Y ** 2).sum()
                        t = []
                        for block, indices in zip(range(self.num_blocks_), feature_indices):
                            # Calculate block scores t1, t2, ... as tn = Xn*wn
                            t.append(X[:, indices[0]:indices[1]].dot(self.W_[block][:, component:component+1]))
                            self.T_[block] = np.hstack((self.T_[block], t[block]))

                        #############################
                        # Calculate explained part of X matrix
                        X_explained = self.Ts_[:, component:component + 1].dot(
                            self.P_[:, component:component + 1].T)

                        varx_explained = (X_explained ** 2).sum()
                        self.explained_var_x_.append(varx_explained / varx)

                        varx_blocks_explained = []
                        if component == 0:
                            varxblocks = []
                        for indices, block in zip(feature_indices, range(self.num_blocks_)):
                            if component == 0:
                                varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                            varx_explained = (X_explained[:, indices[0]:indices[1]] ** 2).sum()
                            varx_blocks_explained.append(varx_explained / varxblocks[block])
                        self.explained_var_xblocks_ = np.hstack(
                            (self.explained_var_xblocks_, np.matrix(varx_blocks_explained).T))

                        # Deflate X matrix
                        X = X - X_explained

                        # 9. Calculate explained variance in Y
                        Y_explained = self.Ts_[:, component:component + 1].dot(self.V_[:, component:component+1].T)
                        vary_explained = (Y_explained ** 2).sum()
                        self.explained_var_y_.append(vary_explained / vary)

                # restore blockwise loadings
                self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]

            if num_features > num_samples:
                # Based on
                # [1] S. Rännar, F. Lindgren, P. Geladi, and S. Wold, “A PLS kernel algorithm for data sets with many
                # variables and fewer objects. Part 1: Theory and algorithm,” J. Chemom., vol. 8, no. 2, pp. 111–125, Mar. 1994.
                # and
                # [1] S. Rännar, P. Geladi, F. Lindgren, and S. Wold, “A PLS kernel algorithm for data sets with many
                # variables and few objects. Part II: Cross‐validation, missing data and examples,” J. Chemom., vol. 9, no. 6, pp. 459–470, 1995.
                # Calculate association matrices
                AS_X = X.dot(X.T)
                AS_Y = Y.dot(Y.T)
                # Calculate kernel matrix once
                S = AS_X.dot(AS_Y)

                for comp in range(self.n_components):

                    # Calculate the eigenvector with the largest eigenvalue of the kernel matrix
                    if self.full_svd:
                        ts = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]
                    else:
                        ts = svds(S, k=1)[0]

                    # Calculate response-score vector
                    u = AS_Y.dot(ts)
                    u = u / np.linalg.norm(u)

                    # Deflate association and kernel matrices
                    deflate_matrix = np.eye(S.shape[0], S.shape[0])-ts.dot(ts.T)
                    AS_X = deflate_matrix.dot(AS_X).dot(deflate_matrix)
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

                        # IDEA: Check if this could be done more efficiently
                        if component == 0:
                            varx = (X ** 2).sum()
                            vary = (Y ** 2).sum()
                        self.A_ = np.hstack((self.A_, np.matrix(a).T))
                        for block in range(self.num_blocks_):
                            self.W_[block] = np.hstack((self.W_[block], w[block]))
                            self.T_[block] = np.hstack((self.T_[block], t[block]))

                        #############################
                        # Calculate explained part of X matrix
                        X_explained = self.Ts_[:, component:component + 1].dot(
                            self.P_[:, component:component + 1].T)

                        varx_explained = (X_explained ** 2).sum()
                        self.explained_var_x_.append(varx_explained / varx)

                        varx_blocks_explained = []
                        if component == 0:
                            varxblocks = []
                        for indices, block in zip(feature_indices, range(self.num_blocks_)):
                            if component == 0:
                                varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                            varx_explained = (X_explained[:, indices[0]:indices[1]] ** 2).sum()
                            varx_blocks_explained.append(varx_explained / varxblocks[block])
                        self.explained_var_xblocks_ = np.hstack(
                            (self.explained_var_xblocks_, np.matrix(varx_blocks_explained).T))

                        # Deflate X matrix
                        X = X - X_explained

                        # 9. Calculate explained variance in Y
                        Y_explained = self.Ts_[:, component:component + 1].dot(self.V_[:, component:component+1].T)
                        vary_explained = (Y_explained ** 2).sum()
                        self.explained_var_y_.append(vary_explained / vary)

                        # 10. Upweight Block Importances of blocks with less features
                        sum_vars = []
                        for vector in w:
                            sum_vars.append(len(vector))
                        if len(a) == 1:
                            a_corrected = [1]
                        else:
                            a_corrected = []
                            for bip, sum_var in zip(a, sum_vars):
                                factor = 1 - sum_var / np.sum(sum_vars)
                                a_corrected.append(bip * factor)
                            a_corrected = list(a_corrected / np.sum(a_corrected))
                        self.A_corrected_ = np.hstack((self.A_corrected_, np.matrix(a_corrected).T))

                # restore blockwise loadings
                self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]

            return self

        elif self.method == 'NIPALS':
            # Restore X blocks (for each deflation step)
            Xblocks = []
            for indices in feature_indices:
                Xblocks.append(X[:, indices[0]:indices[1]])
            Y_calc = Y
            # Wangen and Kowalski (1988)
            for comp in range(self.n_components):
                if self.calc_all:
                    # Save variance of matrices
                    if comp == 0:
                        varx = (X ** 2).sum()
                        vary = (Y ** 2).sum()
                # 0. Take first column vector out of y and regress against each block
                u_a = Y_calc[:, 0:1]
                run = 1
                diff_t = 1
                while diff_t > self.max_tol:  # Condition on error of ts
                    # 1. Regress u_a against all blocks
                    weights = []
                    weights_non_normal = []
                    for block in range(self.num_blocks_):
                        weights.append(Xblocks[block].T.dot(u_a) / u_a.T.dot(u_a))
                        weights_non_normal.append(Xblocks[block].T.dot(u_a) / u_a.T.dot(u_a))
                        # normalize block weigths
                        weights[block] = weights[block] / np.linalg.norm(weights[block])
                    # 2. Regress block weights against rows of each block
                    scores = []
                    for block in range(self.num_blocks_):
                        # Diverging from Wangen and Kowalski by using regression instead of dividing by number of components
                        scores.append(Xblocks[block].dot(weights[block]))
                    # 3. Append all block scores in T_
                    T_ = np.hstack((scores))
                    # 4. Regress u_a against block of block scores
                    superweights = T_.T.dot(u_a) / u_a.T.dot(u_a)
                    superweights = superweights / np.linalg.norm(superweights)
                    # 5. Regress superweights against T_ to obtain superscores
                    superscores = T_.dot(superweights)
                    superscores = superscores / np.linalg.norm(superscores)
                    if run == 1:
                        pass
                    else:
                        diff_t = np.sum(superscores_old - superscores)
                    superscores_old = np.copy(superscores)
                    # 6. Regress superscores agains Y_calc
                    response_weights = Y_calc.T.dot(superscores) / superscores.T.dot(superscores)
                    # 7. Regress response_weights against Y
                    response_scores = Y_calc.dot(response_weights) / response_weights.T.dot(response_weights)
                    response_scores = response_scores / np.linalg.norm(response_scores)
                    u_a = response_scores
                    run += 1

                # 8. Calculate loading
                loadings = [None] * self.num_blocks_
                for block in range(self.num_blocks_):
                    loadings[block] = Xblocks[block].T.dot(superscores)
                # Concatenate the block-loadings
                loadings_total = np.vstack(loadings)

                if self.calc_all:
                    varx_explained = (superscores.dot(loadings_total.T) ** 2).sum()
                    self.explained_var_x_.append(varx_explained / varx)
                    vary_explained = (superscores.dot(response_weights.T) ** 2).sum()
                    self.explained_var_y_.append(vary_explained / vary)
                    varx_blocks_explained = []
                    if comp == 0:
                        varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks_)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (superscores.dot(loadings[block].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])
                    self.explained_var_xblocks_ = np.hstack(
                        (self.explained_var_xblocks_, np.matrix(varx_blocks_explained).T))
                    # 10. Upweight Block Importances of blocks with less features (provided as additional figure of merit)
                    sum_vars = []
                    a = superweights ** 2
                    for vector in weights:
                        sum_vars.append(len(vector))
                    if len(a) == 1:
                        a_corrected = [1]
                    else:
                        a_corrected = []
                        for bip, sum_var in zip(a, sum_vars):
                            factor = 1 - sum_var / np.sum(sum_vars)
                            a_corrected.append(bip * factor)
                        a_corrected = list(a_corrected / np.sum(a_corrected))
                    self.A_corrected_ = np.hstack((self.A_corrected_, np.matrix(a_corrected)))


                # 9. Deflate X_calc
                for block in range(self.num_blocks_):
                    Xblocks[block] = Xblocks[block] - superscores.dot(loadings[block].T)

                # 10. Deflate Y - No deflation of Y
                #Y_calc = Y_calc - superscores.dot(response_weights.T)

                # 11. Append the resulting vectors
                self.V_ = np.hstack((self.V_, response_weights))
                self.U_ = np.hstack((self.U_, response_scores))
                self.A_ = np.hstack((self.A_, superweights ** 2))  # squared for total length 1
                self.Ts_ = np.hstack((self.Ts_, superscores))
                self.P_ = np.hstack((self.P_, loadings_total))
                for block in range(self.num_blocks_):
                    self.W_[block] = np.hstack((self.W_[block], weights[block]))
                    self.W_non_normal_[block] = np.hstack((self.W_non_normal_[block], weights_non_normal[block]))
                    self.T_[block] = np.hstack((self.T_[block], scores[block]))

            # Concatenate weights for beta_ calculation
            weights_total = np.concatenate((self.W_non_normal_), axis=0)
            weights_total = weights_total / np.linalg.norm(weights_total, axis=0)
            self.R_ = weights_total.dot(np.linalg.pinv(self.P_.T.dot(weights_total)))
            self.beta_ = self.R_.dot(self.V_.T)
            # restore blockwise loadings
            self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]

            return self

        elif self.method == 'SIMPLS':
            from warnings import warn
            warn("Method 'SIMPLS' does not calculate A_ and T_!")
            # de Jong 1993
            S = X.T.dot(Y)
            for comp in range(self.n_components):
                if self.full_svd:
                    q = np.linalg.svd(S.T.dot(S), full_matrices=self.full_svd)[0][:, 0:1]
                else:
                    q = svds(S.T.dot(S), k=1)[0]
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
            # restore blockwise loadings
            self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]
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
            of arrays containing all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
        (optional) Y : array
            1-dim or 2-dim array of reference values

        Returns
        ----------
        Super_scores : np.array

        Block_scores : list
        List of np.arrays containing the block scores

        Y_scores : np.array (optional)
        Y-scores, if y was given
        """
        check_is_fitted(self, 'beta_')

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

            X_comp = np.hstack(X)

            Ts_ = X_comp.dot(self.R_)

            if Y is not None:
                Y = check_array(Y, dtype=np.float64, ensure_2d=False)
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                Y = self.y_scaler_.transform(Y)
                # Here the block scores are calculated iteratively for new blocks
                if self.method is not 'SIMPLS': 
                    T_ = []
                    for block in range(self.num_blocks_):
                        T_.append(np.empty((X[block].shape[0], 0)))
                        for comp in range(self.n_components):
                            if comp == 0:
                                T_[block] = X[block].dot(self.W_[block][:, comp:comp+1])
                            else:
                                # deflate the block
                                X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                                T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp+1])))

                    return Ts_, T_ , Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
                else:
                    return Ts_, Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
            else:
                if self.method is not 'SIMPLS':
                    # Here the block scores are calculated iteratively for new blocks
                    T_ = []
                    for block in range(self.num_blocks_):
                        T_.append(np.empty((X[block].shape[0], 0)))
                        for comp in range(self.n_components):
                            if comp == 0:
                                T_[block] = X[block].dot(self.W_[block][:, comp:comp + 1])
                            else:
                                # deflate the block
                                X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                                T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp + 1])))
                    return Ts_, T_
                else:
                    return Ts_

        else:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64)
            else:
                # Check dimensions
                X = [check_array(X, dtype=np.float64)]

            X_comp = np.hstack(X)

            Ts_ = X_comp.dot(self.R_)

            if Y is not None:
                Y = check_array(Y, dtype=np.float64, ensure_2d=False)
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                    # Here the block scores are calculated iteratively for new blocks
                    T_ = []
                    for block in range(self.num_blocks_):
                        T_.append(np.empty((X[block].shape[0], 0)))
                        for comp in range(self.n_components):
                            if comp == 0:
                                T_[block] = X[block].dot(self.W_[block][:, comp:comp + 1])
                            else:
                                # deflate the block
                                X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                                T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp + 1])))
                return Ts_, T_, Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0), T_
            else:
                # Here the block scores are calculated iteratively for new blocks
                T_ = []
                for block in range(self.num_blocks_):
                    T_.append(np.empty((X[block].shape[0], 0)))
                    for comp in range(self.n_components):
                        if comp == 0:
                            T_[block] = X[block].dot(self.W_[block][:, comp:comp + 1])
                        else:
                            # deflate the block
                            X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                            T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp + 1])))
                return Ts_, T_

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

    def plot(self, num_components=2):
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
            #fig = plt.figure()
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
                # IDEA: Use loadings instead of weights
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
