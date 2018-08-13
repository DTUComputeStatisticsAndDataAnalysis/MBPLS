#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:31:48 2018

@authors: Andreas Baum, andba@dtu.dk; Laurent Vermue, lauve@dtu.dk

"""

from .base import BaseEstimator, AdditionalMethods
import collections
import numpy as np
from sklearn.preprocessing import StandardScaler

__all__ = ['MBPLS']


class MBPLS(BaseEstimator, AdditionalMethods):
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

    def __init__(self, n_components, method='SVD', full_svd=True, standardize=True):
        self.n_components = n_components
        self.full_svd = full_svd
        self.method = method
        self.standardize = standardize
        self.max_tol = 1e-14
        self.fitted = False

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

        global U, T, R
        if self.standardize:
            self.x_scalers = []
            if isinstance(X, list):
                for block in range(len(X)):
                    self.x_scalers.append(StandardScaler(with_mean=True, with_std=True))
                    X[block] = self.x_scalers[block].fit_transform(X[block])
            else:
                raise AttributeError("The different blocks have to be passed in a list")

            self.y_scaler = StandardScaler(with_mean=True, with_std=True)
            Y = self.y_scaler.fit_transform(Y)

        self.num_blocks = len(X)

        # Store start/end feature indices for all x blocks
        feature_indices = []
        for block in X:
            if len(feature_indices) > 0:
                feature_indices.append(np.array([feature_indices[-1][1], feature_indices[-1][1] + block.shape[1]]))
            else:
                feature_indices.append(np.array([0, block.shape[1]]))

        self.W = []
        self.W_non_normal = []
        self.T = []
        self.A = np.empty((self.num_blocks, 0))
        self.A_corrected = np.empty((self.num_blocks, 0))
        self.explained_var_xblocks = np.empty((self.num_blocks, 0))
        self.V = np.empty((Y.shape[1], 0))
        self.loading_y = np.empty((Y.shape[1], 0))
        self.U = np.empty((Y.shape[0], 0))
        self.Ts = np.empty((Y.shape[0], 0))
        self.explained_var_y = []
        self.explained_var_x = []

        for block in range(self.num_blocks):
            self.W.append(np.empty((X[block].shape[1], 0)))
            self.W_non_normal.append(np.empty((X[block].shape[1], 0)))
            self.T.append(np.empty((X[block].shape[0], 0)))

        # Concatenate X blocks
        X = np.hstack(X)
        self.P = np.empty((X.shape[1], 0))
        weights = np.empty((X.shape[1], 0))

        if self.method == 'SVD':
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
                    # IDEA: Norming of vectors
                    eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]

                    # 3. Calculate block loadings w1, w2, ... , superweights a1, a2, ...
                    w = []
                    a = []
                    for indices, block in zip(feature_indices, range(self.num_blocks)):
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

                    # 8. Deflate X and calculate explained variance in Xtotol; X1, X2, ... Xk
                    p = np.dot(X.T, ts) / np.dot(ts.T, ts)
                    varx_explained = (np.dot(ts, p.T) ** 2).sum()
                    if comp == 0: varx = (X ** 2).sum()
                    self.explained_var_x.append(varx_explained / varx)

                    varx_blocks_explained = []
                    if comp == 0: varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (np.dot(ts, p[indices[0]:indices[1]].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])

                    X = X - np.dot(ts, p.T)

                    # 9. Calculate explained variance in Y
                    vary_explained = (np.dot(ts, v.T) ** 2).sum()
                    vary = (Y ** 2).sum()
                    self.explained_var_y.append(vary_explained / vary)

                    # 10. Upweight Block Importances of blocks with less features
                    sum_vars = []
                    for vector in w:
                        sum_vars.append(len(vector))
                    a_corrected = []
                    for bip, sum_var in zip(a, sum_vars):
                        factor = 1 - sum_var / np.sum(sum_vars)
                        a_corrected.append(bip * factor)
                    a_corrected = list(a_corrected / np.sum(a_corrected))

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T, W, U, V, Ts, weights, P and A
                    self.V = np.hstack((self.V, v))
                    self.U = np.hstack((self.U, u))
                    self.A = np.hstack((self.A, np.matrix(a).T))
                    self.A_corrected = np.hstack((self.A_corrected, np.matrix(a_corrected).T))
                    self.explained_var_xblocks = np.hstack(
                        (self.explained_var_xblocks, np.matrix(varx_blocks_explained).T))
                    self.Ts = np.hstack((self.Ts, ts))
                    self.P = np.hstack((self.P, p))
                    weights = np.hstack((weights, eigenv))
                    for block in range(self.num_blocks):
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
                    for indices, block in zip(feature_indices, range(self.num_blocks)):
                        partialloading = eigenv[indices[0]:indices[1]]
                        w.append(partialloading / np.linalg.norm(partialloading))
                        a.append(np.linalg.norm(partialloading) ** 2)

                    # 7. Calculate block scores t1, t2, ... as tn = Xn*wn
                    t = []
                    for block, blockloading in zip(Xblocks, w):
                        t.append(np.dot(block, blockloading))

                    # 8. Deflate X and calculate explained variance in Xtotol; X1, X2, ... Xk
                    p = np.dot(X.T, ts) / np.dot(ts.T, ts)
                    varx_explained = (np.dot(ts, p.T) ** 2).sum()
                    # TODO: Clear purpose of this line
                    if comp == 0: varx = (X ** 2).sum()
                    self.explained_var_x.append(varx_explained / varx)

                    varx_blocks_explained = []
                    if comp == 0: varxblocks = []
                    for indices, block in zip(feature_indices, range(self.num_blocks)):
                        if comp == 0:
                            varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (np.dot(ts, p[indices[0]:indices[1]].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])

                    X = X - np.dot(ts, p.T)

                    # 9. Calculate explained variance in Y
                    vary_explained = (np.dot(ts, v.T) ** 2).sum()
                    vary = (Y ** 2).sum()
                    self.explained_var_y.append(vary_explained / vary)

                    # 10. Upweight Block Importances of blocks with less features (provided as additional figure of merit)
                    sum_vars = []
                    for vector in w:
                        sum_vars.append(len(vector))
                    a_corrected = []
                    for bip, sum_var in zip(a, sum_vars):
                        factor = 1 - sum_var / np.sum(sum_vars)
                        a_corrected.append(bip * factor)
                    a_corrected = list(a_corrected / np.sum(a_corrected))

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T, W, U, V, Ts, weights, P and A
                    self.V = np.hstack((self.V, v))
                    self.U = np.hstack((self.U, u))
                    self.A = np.hstack((self.A, np.matrix(a).T))
                    self.A_corrected = np.hstack((self.A_corrected, np.matrix(a_corrected).T))
                    self.explained_var_xblocks = np.hstack(
                        (self.explained_var_xblocks, np.matrix(varx_blocks_explained).T))
                    self.Ts = np.hstack((self.Ts, ts))
                    self.P = np.hstack((self.P, p))
                    weights = np.hstack((weights, eigenv))
                    for block in range(self.num_blocks):
                        self.W[block] = np.hstack((self.W[block], w[block]))
                        self.T[block] = np.hstack((self.T[block], t[block]))
                    # TODO: implement R coefficient
                    pseudoinv = np.dot(weights, np.linalg.pinv(np.dot(self.P.T, weights)))
                    pseudoinv = np.dot(pseudoinv, np.linalg.pinv(np.dot(self.Ts.T, self.Ts)))
                    pseudoinv = np.dot(pseudoinv, self.Ts.T)
                    self.beta = np.dot(pseudoinv, Y)

            self.fitted = True

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
                    for block in range(self.num_blocks):
                        weights.append(np.dot(Xblocks[block].T, u_a) / np.dot(u_a.T, u_a))
                        weights_non_normal.append(np.dot(Xblocks[block].T, u_a) / np.dot(u_a.T, u_a))
                        # normalize block weigths
                        weights[block] = weights[block] / np.linalg.norm(weights[block])
                    # 2. Regress block weights against rows of each block
                    scores = []
                    for block in range(self.num_blocks):
                        # Diverging from Wangen and Kowalski by using regression instead of dividing by number of components
                        scores.append(np.dot(Xblocks[block], weights[block]) / np.dot(weights[block].T, weights[block]))
                    # 3. Append all block scores in T
                    T = np.hstack((scores))
                    # 4. Regress u_a against block of block scores
                    superweights = np.dot(T.T, u_a) / np.dot(u_a.T, u_a)
                    superweights = superweights / np.linalg.norm(superweights)
                    # 5. Regress superweights against T to obtain superscores
                    superscores = np.dot(T, superweights) / np.dot(superweights.T, superweights)
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
                loadings = [None] * self.num_blocks
                for block in range(self.num_blocks):
                    loadings[block] = np.dot(Xblocks[block].T, superscores) / np.dot(superscores.T, superscores)
                loading_y = Y_calc.T.dot(response_scores) / response_scores.T.dot(response_scores)
                # 9. Deflate X_calc
                for block in range(self.num_blocks):
                    Xblocks[block] = Xblocks[block] - np.dot(superscores, loadings[block].T)

                # 10. Deflate Y - No deflation of Y
                # Y_calc = Y_calc - np.dot(superscores, response_weights.T)

                # 11. Append the resulting vectors
                self.V = np.hstack((self.V, response_weights))
                self.loading_y = np.hstack((self.loading_y, loading_y))
                self.U = np.hstack((self.U, response_scores))
                # IDEA: Shouldn't this be the quotient
                # superweights = superweights / np.sum(superweights, axis=0)
                # self.A = np.hstack((self.A, superweights / np.sum(superweights, axis=0)))
                self.A = np.hstack((self.A, superweights ** 2))  # squared for total length 1
                self.Ts = np.hstack((self.Ts, superscores))
                # Concatenate the block-loadings
                loadings = np.vstack(loadings)
                self.P = np.hstack((self.P, loadings))
                for block in range(self.num_blocks):
                    self.W[block] = np.hstack((self.W[block], weights[block] * -1))
                    self.W_non_normal[block] = np.hstack((self.W_non_normal[block], weights_non_normal[block] * -1))
                    self.T[block] = np.hstack((self.T[block], scores[block] * -1))

            # Negate results to achieve same results as with SVD
            self.Ts *= -1
            self.P *= -1
            self.U *= -1
            self.V *= -1
            self.loading_y *= -1

            weights_total = np.concatenate((self.W_non_normal), axis=0)  # Concatenate weights for beta calculation
            weights_total = weights_total / np.linalg.norm(weights_total, axis=0)
            pseudoinv = np.linalg.pinv((np.dot(self.P.T, weights_total)))
            R = np.dot(weights_total, pseudoinv)
            self.R = R
            self.beta = np.dot(R, self.V.T)

            # Testing R_y
            # weights_y_norm = self.V / np.linalg.norm(self.V, axis=0)
            # pseudoinv_y = np.linalg.pinv((np.dot(self.loading_y.T, weights_y_norm)))
            # R_y = np.dot(weights_y_norm, pseudoinv_y)
            # self.R_y = R_y

            self.fitted = True
            return self

        elif self.method == 'SIMPLS':
            # de Jong 1993
            S = np.dot(X.T, Y)
            for comp in range(self.n_components):
                q = np.linalg.svd(S.T.dot(S), full_matrices=self.full_svd)[0][:, 0:1]
                r = S.dot(q)
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
                    v = v - V.dot(V.T.dot(p))
                    u = u - T.dot(T.T.dot(u))
                v = v / np.sqrt(v.T.dot(v))
                S = S - v.dot(v.T.dot(S))
                # Calculation of block importance
                # TODO: This calculation is not correct
                score_block = np.empty((X.shape[0], 0))
                for indices in feature_indices:
                    score_vector = X[:, indices[0]:indices[1]].dot(r[indices[0]:indices[1]])
                    score_block = np.hstack((score_block, score_vector))
                superweights = score_block.T.dot(u) / u.T.dot(u)
                superweights = superweights / np.linalg.norm(superweights)


                if comp == 0:
                    R = r
                    T = t
                    P = p
                    Q = q
                    U = u
                    V = v
                    A = superweights ** 2  # squared for total length 1
                    Ts = score_block
                else:
                    R = np.hstack((R, r))
                    T = np.hstack((T, t))
                    P = np.hstack((P, p))
                    Q = np.hstack((Q, q))
                    U = np.hstack((U, u))
                    V = np.hstack((V, v))
                    A = np.hstack((A, superweights ** 2))  # squared for total length 1

            self.P = P
            self.A = A
            self.Ts = T
            self.U = U / np.linalg.norm(U)
            self.R = R
            self.beta = R.dot(Q.T)
            self.T = Ts



            self.fitted = True
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
        assert self.fitted == True, 'The model has not been fitted yet'
        assert isinstance(X, list), "The different blocks have to be passed in a list"
        # IDEA: Return scores per Block

        if self.standardize:
            for block in range(len(X)):
                X[block] = self.x_scalers[block].transform(X[block])

            X = np.hstack(X)
            return X.dot(self.R)

        else:
            X = np.hstack(X)
            return X.dot(self.R)

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
        assert self.fitted == True, 'The model has not been fitted yet'

        if self.standardize:
            if isinstance(X, list):
                for block in range(len(X)):
                    X[block] = self.x_scalers[block].transform(X[block])
            else:
                raise AttributeError("The different blocks have to be passed in a list")
            X = np.hstack(X)
            y_hat = self.y_scaler.inverse_transform(X.dot(self.beta))
        else:
            X = np.hstack(X)
            y_hat = X.dot(self.beta)

        return y_hat

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
        plt.close('all')

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

            gs1 = GridSpec(1, self.num_blocks, top=0.875, bottom=0.85, right=0.95)
            for block in range(self.num_blocks):
                plt.subplot(gs1[0, block])
                plt.text(0.5, 0.5, "X-Block {:d}\nImportance: {:.0f}%".format(block + 1, self.A[block, comp] * 100),
                         fontsize=12, horizontalalignment='center')
                plt.axis('off')

            gs2 = GridSpec(2, self.num_blocks, top=0.8, hspace=0.45, wspace=0.45, right=0.95)
            loading_axes = []
            score_axes = []
            # List for inverse transforming the loadings/weights
            W_inv_trans = []
            for block in range(self.num_blocks):
                # Inverse transforming weights/loadings
                # TODO: Does this make sense?
                if self.standardize:
                    W_inv_trans.append(self.x_scalers[block].inverse_transform(self.W[block][:, comp]))
                else:
                    W_inv_trans.append(self.W[block])

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
                    plt.plot(self.T[block][:, comp])
                    step = int(self.T[block].shape[0] / 4)
                    plt.xticks(np.arange(0, self.T[block].shape[0], step),
                               np.arange(1, self.T[block].shape[0] + 1, step))
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
                    plt.plot(self.T[block][:, comp])
                    step = int(self.T[block].shape[0] / 4)
                    plt.xticks(np.arange(0, self.T[block].shape[0], step),
                               np.arange(1, self.T[block].shape[0] + 1, step))
                    # plt.setp(score_axes[block].get_yticklabels(), visible=False)
                    plt.xlabel("Sample")
                    plt.grid()

            plt.show()

        plt.suptitle("Block importances", fontsize=14, fontweight='bold')
        gs3 = GridSpec(1, 1, top=0.825, right=0.7, hspace=0.45, wspace=0.4)
        ax = plt.subplot(gs3[0, 0])
        width = 0.8 / len(num_components)
        for i, comp in enumerate(num_components):
            ax.bar(np.arange(self.num_blocks) + 0.6 + i * width, 100 * np.ravel(self.A[:, comp]), width=width, \
                   label="Component {}".format(comp + 1))
        ax.set_xticklabels(list(np.arange(self.num_blocks) + 1))
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.4))
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_xlabel("Block")
        ax.set_ylabel("Block importance in %")
        plt.show()
