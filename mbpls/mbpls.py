#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author:   Andreas Baum <andba@dtu.dk>
#           Laurent Vermue <lauve@dtu.dk>
#
# License: 3-clause BSD


from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, MultiOutputMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_consistent_length
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds
from warnings import warn

__all__ = ['MBPLS']


class MBPLS(BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin):
    """(Multiblock) PLS regression for predictive modelling using latent variables
        --------------------------------------------------------------------------
        
        - **PLS1**: Predict a response vector :math:`y` from a single multivariate data block :math:`X`
        - **PLS2**: Predict a response matrix :math:`Y` from a single multivariate data block :math:`X`
        - **MBPLS**: Predict a response vector/matrix :math:`Y` from multiple data blocks :math:`X_1, X_2, ... , X_i`
        
        
        
        for detailed information check [ref]
        
        
        Model settings
        --------------

        method : string (default 'NIPALS')
            The method being used to derive the model attributes, possible are 'UNIPALS', 'NIPALS', 'SIMPLS' and
            'KERNEL'

        n_components : int
            Number (:math:`k`) of Latent Variables (LV)

        standardize : bool (default True)
            Standardizing the data

        full_svd : bool (default True)
            Using full singular value decomposition when performing SVD method. 
            Set to 'False' when using very large quadratic matrices :math:`X`.

        max_tol : non-negative float (default 1e-14)
            Maximum tolerance allowed when using the iterative NIPALS algorithm

        calc_all : bool (default True)
            Calculate all internal attributes for the used method. Some methods do not need to calculate all attributes,
            i.e. scores, weights etc., to obtain the regression coefficients used for prediction. Setting this parameter
            to false will omit these calculations for efficiency and speed.

        sparse_data : bool (default False)
            NIPALS is the only algorithm that can handle sparse data using the method of H. Martens and Martens (2001)
            (p. 381). If this parameter is set to 'True', the method will be forced to NIPALS and sparse data is
            allowed.
            Without setting this parameter to 'True', sparse data will not be accepted.

        
        Model attributes after fitting
        ------------------------------

        **X-side** 

        Ts_ : array, super scores :math:`[n,k]`
        
        T_ : list, block scores :math:`[i][n,k]`
        
        W_ : list, block weights :math:`[i][p_i,k]`
        
        A_ : array, block importances/super weights :math:`[i,k]`
        
        A_corrected_ : array, normalized block importances :math:`A_{corr,ik} = A_{ik} \\cdot (1- \\frac{p_i}{p})`
        
        P_ : list, block loadings :math:`[i][p_i,k]`
        
        R_ : array, x_rotations :math:`R = W (P^T W)^{-1}`
        
        explained_var_x_ : list, explained variance in :math:`X` per LV :math:`[k]`
        
        explained_var_xblocks_ : array, explained variance in each block :math:`X_i` :math:`[i,k]`
        
        beta_ : array, regression vector :math:`\\beta`  :math:`[p,q]`
        

        **Y-side**
        
        U_ : array, scoresInitialize :math:`[n,k]`
        
        V_ : array, loadings :math:`[q,k]`
        
        explained_var_y_ : list, explained variance in :math:`Y` :math:`[k]`
        
        

        Notes
        -----
        
        According to literature one distinguishes between PLS1 [ref], PLS2 [ref] and MBPLS [ref].
        Common goal is to find loading vectors :math:`p` and :math:`v` which project the data to latent variable scores
        :math:`ts` and :math:`u` indicating maximal covariance. Subsequently, the explained variance is deflated and
        further LVs can be extracted. Deflation for the :math:`k`-th LV is obtained as:
           
        
        .. math::
            
            X_{k+1} = X_{k} - t_k p_k^T
            
        
        **PLS1**: Matrices are computed such that:
            
        .. math::
                        
            X &= T_s P^T + E_X
            
            y &= X \\beta + e
        
        **PLS2**: Matrices are computed such that:
            
        .. math::
            
            X &= T_s P^T + E_X
            
            Y &= U V^T + E_Y
            
            Y &= X \\beta + E


        **MBPLS**: In addition, MBPLS provides a measure for how important (:math:`a_{ik}`) each block :math:`X_i` is
        for prediction of :math:`Y` in the :math:`k`-th LV. Matrices are computed such that:

        .. math::

            X &= [X_1|X_2|...|X_i]

            X_i &= T_s P_i ^T + E_i

            Y &= U V^T + E_Y

            Y &= X \\beta + E

        using the following calculation:

        :math:`X_k = X`

        for k in K:

        .. math::

              w_{k} &= \\text{first eigenvector of } X_k^T Y Y^T X_k, ||w_k||_2 = 1

              w_{k} &= [w_{1k}|w_{2k}|...|w_{ik}]

              a_{ik} &= ||w_{ik}||_2 ^2

              t_{ik} &= \\frac{X_i w_{ik}}{||w_{ik}||_2}

              t_{sk} &= \\sum{a_{ik} * t_{ik}}

              v_k &= \\frac{Y^T t_{sk}}{t_{sk} ^T t_{sk}}

              u_k &= Y v_k

              u_k &= \\frac{u_k}{||u_k||_2}

              p_k &= \\frac{X^T t_{sk}}{t_{sk} ^T t_{sk}}, p_k = [p_{1k}|p_{2k}|...|p_{ik}]

              X_{k+1} &= X_k - t_{sk} p_k

        End loop

        :math:`P = [p_{1}|p_{2}|...|p_{K}]`

        :math:`T_{s} = [t_{s1}|t_{s2}|...|t_{sK}]`

        :math:`U = [u_{1}|u_{2}|...|u_{K}]`

        :math:`V = [v_{1}|v_{2}|...|v_{K}]`

        :math:`W = [w_{1}|w_{2}|...|w_{k}]`

        :math:`R = W (P^T W)^{-1}`

        :math:`\\beta = R V^T`
               
        
        Examples
        --------
        
        Quick Start: Two random data blocks :math:`X_1` and :math:`X_2` and a random reference vector :math:`y` for
        predictive modeling.
        
        .. code-block:: python
        
            import numpy as np
            from mbpls.mbpls import MBPLS
            
            mbpls = MBPLS(n_components=4)
            x1 = np.random.rand(20,300)
            x2 = np.random.rand(20,450)
            
            y = np.random.rand(20,1)
            
            mbpls.fit([x1, x2],y)
            mbpls.plot(num_components=4)
            
            y_pred = mbpls.predict([x1, x2])
            
        More elaborate examples can be found at
        https://github.com/DTUComputeStatisticsAndDataAnalysis/MBPLS/tree/master/examples
    """

    def __init__(self, n_components=2, full_svd=False, method='NIPALS', standardize=True, max_tol=1e-14, calc_all=True,
                 sparse_data=False):
        self.n_components = n_components
        self.full_svd = full_svd
        self.method = method
        self.standardize = standardize
        self.max_tol = max_tol
        self.calc_all = calc_all
        self.sparse_data = sparse_data

    def check_sparsity_level(self, data):
        total_rows, total_columns = data.shape
        sparse_columns = np.isnan(data).sum(axis=0) > 0
        sparse_columns_n = sparse_columns.sum()
        dense_columns = np.where(sparse_columns == 0)[0]
        sparse_columns = np.where(sparse_columns == 1)[0]
        sparse_rows = np.isnan(data).sum(axis=1) > 0
        sparse_rows_n = sparse_rows.sum()
        dense_rows = np.where(sparse_rows == 0)[0]
        sparse_rows = np.where(sparse_rows == 1)[0]
        if sparse_columns_n/total_columns > 0.5:
            warn("The sparsity of your data is likely to high for this algorithm. This can cause either convergence"
                 "problems or crash the algorithm.")
        if sparse_rows_n/total_rows > 0.5:
            warn("The sparsity of your data is likely to high for this algorithm. This can cause either convergence"
                 "problems or crash the algorithm.")
        return sparse_rows, sparse_columns, dense_rows, dense_columns

    def fit(self, X, Y):
        """ Fit model to given data

        Parameters
        ----------

        X : list
            of all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
        Y : array
            1-dim or 2-dim array of reference values
        """

        # In case of sparse data, check if chosen method is suitable
        if self.sparse_data is True:
            if self.method != 'NIPALS':
                warn("The parameter sparse data was set to 'True', but the chosen method is not 'NIPALS'."
                     "The method will be set to 'NIPALS'")
                self.method = 'NIPALS'

        global U_, T_, R_
        Y = check_array(Y, dtype=np.float64, ensure_2d=False, force_all_finite=not self.sparse_data)
        if self.sparse_data is True:
            self.sparse_Y_info_ = {}
            self.sparse_Y_info_['Y'] = self.check_sparsity_level(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if self.standardize:
            self.x_scalers_ = []
            if isinstance(X, list) and not isinstance(X[0], list):
                if self.sparse_data is True:
                    self.sparse_X_info_ = {}
                for block in range(len(X)):
                    self.x_scalers_.append(StandardScaler(with_mean=True, with_std=True))
                    # Check dimensions
                    check_consistent_length(X[block], Y)
                    X[block] = check_array(X[block], dtype=np.float64, copy=True, force_all_finite=not self.sparse_data)
                    if self.sparse_data is True:
                        self.sparse_X_info_[block] = self.check_sparsity_level(X[block])
                    X[block] = self.x_scalers_[block].fit_transform(X[block])
            else:
                self.x_scalers_.append(StandardScaler(with_mean=True, with_std=True))
                # Check dimensions
                X = check_array(X, dtype=np.float64, copy=True, force_all_finite=not self.sparse_data)
                if self.sparse_data is True:
                    self.sparse_X_info_ = {}
                    self.sparse_X_info_[0] = self.check_sparsity_level(X)
                check_consistent_length(X, Y)
                X = [self.x_scalers_[0].fit_transform(X)]

            self.y_scaler_ = StandardScaler(with_mean=True, with_std=True)
            Y = self.y_scaler_.fit_transform(Y)
        else:
            if isinstance(X, list) and not isinstance(X[0], list):
                if self.sparse_data is True:
                    self.sparse_X_info_ = {}
                for block in range(len(X)):
                    # Check dimensions
                    check_consistent_length(X[block], Y)
                    X[block] = check_array(X[block], dtype=np.float64, copy=True, force_all_finite=not self.sparse_data)
                    if self.sparse_data is True:
                        self.sparse_X_info_[block] = self.check_sparsity_level(X[block])
            else:
                # Check dimensions
                X = check_array(X, dtype=np.float64, copy=True, force_all_finite=not self.sparse_data)
                if self.sparse_data is True:
                    self.sparse_X_info_ = {}
                    self.sparse_X_info_[0] = self.check_sparsity_level(X)
                check_consistent_length(X, Y)
                X = [X]

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
                    S = np.linalg.multi_dot([X.T, Y, Y.T, X])
                    if self.full_svd:
                        eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]
                    else:
                        eigenv = svds(S, k=1)[0]

                    # 3. Calculate block weights w1, w2, ... , superweights a1, a2, ...
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

                    # 10. Correct block importance for number of features in blocks
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
                    self.A_ = np.hstack((self.A_, np.array(a).reshape(-1, 1)))
                    self.A_corrected_ = np.hstack((self.A_corrected_, np.array(a_corrected).reshape(-1,1)))
                    self.explained_var_xblocks_ = np.hstack(
                        (self.explained_var_xblocks_, np.array(varx_blocks_explained).reshape(-1, 1)))
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

            if num_features > num_samples:
                for comp in range(self.n_components):
                    # 1. Restore X blocks (for each deflation step)
                    Xblocks = []
                    for indices in feature_indices:
                        Xblocks.append(X[:, indices[0]:indices[1]])

                    # 2. Calculate ts by SVD(XX'YY') --> eigenvector with largest eigenvalue
                    S = np.linalg.multi_dot([X, X.T, Y, Y.T])
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

                    # 6. Calculate block weights w1, w2, ... , superweights a1, a2, ...
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

                    # 10. Correct block importance for number of features in blocks
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
                    self.A_ = np.hstack((self.A_, np.array(a).reshape(-1, 1)))
                    self.A_corrected_ = np.hstack((self.A_corrected_, np.array(a_corrected).reshape(-1, 1)))
                    self.explained_var_xblocks_ = np.hstack(
                        (self.explained_var_xblocks_, np.array(varx_blocks_explained).reshape(-1, 1)))
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
                S = np.linalg.multi_dot([X.T, Y, Y.T, X])
                # Calculate variance covariance matrixes
                VAR = X.T.dot(X)
                COVAR = X.T.dot(Y)
                for comp in range(self.n_components):
                    if self.full_svd:
                        eigenv = np.linalg.svd(S, full_matrices=self.full_svd)[0][:, 0:1]
                    else:
                        eigenv = svds(S, k=1)[0]

                    # 6. Calculate v (Y-loading) by projection of ts on Y
                    denominator = np.linalg.multi_dot([eigenv.T, VAR, eigenv])
                    v = eigenv.T.dot(COVAR) / denominator

                    # 8. Deflate X and calculate explained variance in Xtotal; X1, X2, ... Xk
                    p = eigenv.T.dot(VAR) / denominator

                    if self.calc_all:
                        # 3. Calculate block weights w1, w2, ... , superweights a1, a2, ...
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
                        self.A_corrected_ = np.hstack((self.A_corrected_, np.array(a_corrected).reshape(-1, 1)))
                        u = Y.dot(v.T) / v.dot(v.T)
                        u = u / np.linalg.norm(u)
                        self.U_ = np.hstack((self.U_, u))
                        self.A_ = np.hstack((self.A_, np.array(a).reshape(-1, 1)))

                    # Update kernel and variance/covariance matrices
                    deflate_matrix = np.eye(S.shape[0]) - eigenv.dot(p)
                    S = np.linalg.multi_dot([deflate_matrix.T, S, deflate_matrix])
                    VAR = np.linalg.multi_dot([deflate_matrix.T, VAR, deflate_matrix])
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
                # End
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
                            (self.explained_var_xblocks_, np.array(varx_blocks_explained).reshape(-1, 1)))

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
                # variables and fewer objects. Part 1: Theory and algorithm,” J. Chemom., vol. 8, no. 2, pp. 111–125,
                # Mar. 1994.
                # and
                # [1] S. Rännar, P. Geladi, F. Lindgren, and S. Wold, “A PLS kernel algorithm for data sets with many
                # variables and few objects. Part II: Cross‐validation, missing data and examples,”
                # J. Chemom., vol. 9, no. 6, pp. 459–470, 1995.
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
                    AS_X = np.linalg.multi_dot([deflate_matrix, AS_X, deflate_matrix])
                    AS_Y = np.linalg.multi_dot([deflate_matrix, AS_Y, deflate_matrix])
                    S = AS_X.dot(AS_Y)

                    # 11. add t, w, u, v, ts, eigenv, loading and a to T_, W_, U_, V_, Ts_, weights, P_ and A_
                    self.U_ = np.hstack((self.U_, u))
                    self.Ts_ = np.hstack((self.Ts_, ts))

                self.W_concat_ = X.T.dot(self.U_)
                # Normalize weights to length one per column
                self.W_concat_ = self.W_concat_ / np.linalg.norm(self.W_concat_, axis=0)
                self.P_ = np.linalg.multi_dot([X.T, self.Ts_, np.linalg.pinv(self.Ts_.T.dot(self.Ts_))])
                self.V_ = np.linalg.multi_dot([Y.T, self.Ts_, np.linalg.pinv(self.Ts_.T.dot(self.Ts_))])

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
                        self.A_ = np.hstack((self.A_, np.array(a).reshape(-1, 1)))
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
                            (self.explained_var_xblocks_, np.array(varx_blocks_explained).reshape(-1, 1)))

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
                        self.A_corrected_ = np.hstack((self.A_corrected_, np.array(a_corrected).reshape(-1, 1)))

                # restore blockwise loadings
                self.P_ = np.split(self.P_, [index[1] for index in feature_indices])[:-1]

            return self

        elif self.method == 'NIPALS':
            # Based on [1] J. A. Westerhuis, T. Kourti, and J. F. MacGregor, “Analysis of multiblock and hierarchical
            # PCA and PLS models,” J. Chemom., vol. 12, no. 5, pp. 301–321, Sep. 1998.
            # In case of sparse data, calculations are based on the algorithm of
            # H. Martens and Martens (2001) (p. 381)
            #
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
                        if self.sparse_data:
                            varx = np.nansum(X ** 2)
                            vary = np.nansum(Y ** 2)
                        else:
                            varx = (X ** 2).sum()
                            vary = (Y ** 2).sum()
                # When the data is sparse, the initial vector is not allowed to be sparse
                if self.sparse_data:
                    if len(self.sparse_Y_info_['Y'][1]) == Y_calc.shape[1]:
                        u_a = np.random.rand(Y_calc.shape[0], 1)
                    else:
                        u_a = Y_calc[:, self.sparse_Y_info_['Y'][3][0]:self.sparse_Y_info_['Y'][3][0]+1]
                else: # 0. Take first column vector out of y and regress against each block
                    u_a = Y_calc[:, 0:1]
                run = 1
                diff_t = 1
                while diff_t > self.max_tol:  # Condition on error of ts
                    # 1. Regress u_a against all blocks
                    weights = []
                    weights_non_normal = []
                    for block in range(self.num_blocks_):
                        if self.sparse_data:
                            temp_weights = Xblocks[block].T.dot(u_a) / u_a.T.dot(u_a)
                            for sparse_column in self.sparse_X_info_[block][1]:
                                non_sparse_rows = ~np.isnan(Xblocks[block][:, sparse_column])
                                temp_weights[sparse_column] = \
                                    Xblocks[block][non_sparse_rows, sparse_column].T.dot(u_a[non_sparse_rows]) / \
                                    u_a[non_sparse_rows].T.dot(u_a[non_sparse_rows])
                            weights.append(temp_weights)
                            weights_non_normal.append(temp_weights)
                        else:
                            temp_weights = Xblocks[block].T.dot(u_a) / u_a.T.dot(u_a)
                            weights.append(temp_weights)
                            weights_non_normal.append(temp_weights)
                        # normalize block weigths
                        weights[block] = weights[block] / np.linalg.norm(weights[block])
                    # 2. Regress block weights against rows of each block
                    scores = []
                    for block in range(self.num_blocks_):
                        # Diverging from Wangen and Kowalski by using regression instead of dividing by # of components
                        if self.sparse_data:
                            temp_scores = Xblocks[block].dot(weights[block])
                            for sparse_row in self.sparse_X_info_[block][0]:
                                non_sparse_columns = ~np.isnan(Xblocks[block][sparse_row, :])
                                temp_scores[sparse_row] = \
                                    Xblocks[block][sparse_row, non_sparse_columns].dot(
                                        weights[block][non_sparse_columns]) / \
                                    weights[block][non_sparse_columns].T.dot(weights[block][non_sparse_columns])
                            scores.append(temp_scores)
                        else:
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
                    if self.sparse_data:
                        temp_weights = Y_calc.T.dot(superscores) / superscores.T.dot(superscores)
                        for sparse_column in self.sparse_Y_info_['Y'][1]:
                            non_sparse_rows = ~np.isnan(Y_calc[:, sparse_column])
                            temp_weights[sparse_column] = \
                                Y_calc[non_sparse_rows, sparse_column].T.dot(superscores[non_sparse_rows]) / \
                                superscores[non_sparse_rows].T.dot(superscores[non_sparse_rows])
                        response_weights = temp_weights
                    else:
                        response_weights = Y_calc.T.dot(superscores) / superscores.T.dot(superscores)
                    # 7. Regress response_weights against Y
                    if self.sparse_data:
                        temp_scores = Y_calc.dot(response_weights) / response_weights.T.dot(response_weights)
                        for sparse_row in self.sparse_X_info_[block][0]:
                            non_sparse_columns = ~np.isnan(Y_calc[sparse_row, :])
                            temp_scores[sparse_row] = \
                                Y_calc[sparse_row, non_sparse_columns].dot(response_weights[non_sparse_columns]) / \
                                response_weights[non_sparse_columns].T.dot(response_weights[non_sparse_columns])
                        response_scores = temp_scores / np.linalg.norm(temp_scores)
                        u_a = response_scores
                    else:
                        response_scores = Y_calc.dot(response_weights) / response_weights.T.dot(response_weights)
                        response_scores = response_scores / np.linalg.norm(response_scores)
                        u_a = response_scores
                    run += 1

                # 8. Calculate loading
                loadings = [None] * self.num_blocks_
                for block in range(self.num_blocks_):
                    if self.sparse_data:
                        temp_loadings = Xblocks[block].T.dot(superscores)
                        for sparse_column in self.sparse_X_info_[block][1]:
                            non_sparse_rows = ~np.isnan(Xblocks[block][:, sparse_column])
                            temp_loadings[sparse_column] = \
                                Xblocks[block][non_sparse_rows, sparse_column].T.dot(superscores[non_sparse_rows]) / \
                                superscores[non_sparse_rows].T.dot(superscores[non_sparse_rows])
                        loadings[block] = temp_loadings
                    else:
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
                            if self.sparse_data:
                                varxblocks.append(np.nansum(X[:, indices[0]:indices[1]] ** 2))
                            else:
                                varxblocks.append((X[:, indices[0]:indices[1]] ** 2).sum())
                        varx_explained = (superscores.dot(loadings[block].T) ** 2).sum()
                        varx_blocks_explained.append(varx_explained / varxblocks[block])
                    self.explained_var_xblocks_ = np.hstack(
                        (self.explained_var_xblocks_, np.array(varx_blocks_explained).reshape(-1, 1)))
                    # 10. Upweight Block Importances of blocks with less features
                    # (provided as additional figure of merit)
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
                    self.A_corrected_ = np.hstack((self.A_corrected_, np.array(a_corrected).reshape(-1, 1)))


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
                    v = v - np.linalg.multi_dot([V_, V_.T, p])
                    u = u - np.linalg.multi_dot([T_, T_.T, u])
                v = v / np.sqrt(v.T.dot(v))
                S = S - np.linalg.multi_dot([v, v.T, S])

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

    def transform(self, X, Y=None, return_block_scores=False):
        """ Obtain scores based on the fitted model

         Parameters
        ----------
        X : list
            of arrays containing all xblocks x1, x2, ..., xn. Rows are observations, columns are features/variables
        (optional) Y : array
            1-dim or 2-dim array of reference values
        return_block_scores: bool (default False)
            Returning block scores T_ when transforming the data

        Returns
        ----------
        Super_scores : np.array

        Block_scores : list
        List of np.arrays containing the block scores

        Y_scores : np.array (optional)
        Y-scores, if y was given
        """
        check_is_fitted(self, 'beta_')

        if self.sparse_data:
            sparse_X_info_ = {}
            sparse_Y_info_ = {}

        if self.standardize:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64, force_all_finite=not self.sparse_data)
                    if self.sparse_data:
                        sparse_X_info_[block] = self.check_sparsity_level(X[block])
                    X[block] = self.x_scalers_[block].transform(X[block])
            else:
                # Check dimensions
                X = check_array(X, dtype=np.float64, force_all_finite=not self.sparse_data)
                if self.sparse_data:
                    sparse_X_info_[0] = self.check_sparsity_level(X)
                X = [self.x_scalers_[0].transform(X)]

            X_comp = np.hstack(X)
            if self.sparse_data:
                sparse_X_info_['comp'] = self.check_sparsity_level(X_comp)

            if self.sparse_data:
                temp_Ts = X_comp.dot(self.R_)
                for sparse_row in sparse_X_info_['comp'][0]:
                    non_sparse_columns = ~np.isnan(X_comp[sparse_row, :])
                    temp_Ts[sparse_row] = \
                        X_comp[sparse_row, non_sparse_columns].dot(self.R_[non_sparse_columns])
                Ts_ = temp_Ts
            else:
                Ts_ = X_comp.dot(self.R_)

            if Y is not None:
                Y = check_array(Y, dtype=np.float64, ensure_2d=False, force_all_finite=not self.sparse_data)
                if self.sparse_data:
                    sparse_Y_info_['Y'] = self.check_sparsity_level(Y)
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
                                if self.sparse_data:
                                    temp_scores = X[block].dot(self.W_[block][:, comp:comp+1])
                                    for sparse_row in sparse_X_info_[block][0]:
                                        non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                        temp_scores[sparse_row] = \
                                            X[block][sparse_row, non_sparse_columns].dot(
                                                self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                    T_[block] = temp_scores
                                else:
                                    T_[block] = X[block].dot(self.W_[block][:, comp:comp+1])
                            else:
                                # deflate the block
                                X[block] = X[block] - Ts_[:, comp - 1:comp].dot(self.P_[block][:, comp - 1:comp].T)
                                if self.sparse_data:
                                    temp_scores = X[block].dot(self.W_[block][:, comp:comp + 1])
                                    for sparse_row in sparse_X_info_[block][0]:
                                        non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                        temp_scores[sparse_row] = \
                                            X[block][sparse_row, non_sparse_columns].dot(
                                                self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                    T_[block] = np.hstack((T_[block], temp_scores))
                                else:
                                    T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp+1])))
                    #Calculate Y scores
                    if self.sparse_data:
                        temp_scores = Y.dot(self.V_)
                        for sparse_row in sparse_Y_info_['Y'][0]:
                            non_sparse_columns = ~np.isnan(Y[sparse_row, :])
                            temp_scores[sparse_row] = \
                                Y[sparse_row, non_sparse_columns].dot(self.V_[non_sparse_columns])
                        temp_scores = temp_scores / np.linalg.norm(temp_scores, axis=0)
                        U_ = temp_scores
                    else:
                        U_ = Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
                    if return_block_scores:
                        return Ts_, T_, U_
                    else:
                        return Ts_, U_
                else:
                    # Calculate Y scores
                    if self.sparse_data:
                        temp_scores = Y.dot(self.V_)
                        for sparse_row in sparse_Y_info_['Y'][0]:
                            non_sparse_columns = ~np.isnan(Y[sparse_row, :])
                            temp_scores[sparse_row] = \
                                Y[sparse_row, non_sparse_columns].dot(self.V_[non_sparse_columns])
                        temp_scores = temp_scores / np.linalg.norm(temp_scores, axis=0)
                        U_ = temp_scores
                    else:
                        U_ = Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
                    return Ts_, U_
            else:
                if self.method is not 'SIMPLS':
                    # Here the block scores are calculated iteratively for new blocks
                    T_ = []
                    for block in range(self.num_blocks_):
                        T_.append(np.empty((X[block].shape[0], 0)))
                        for comp in range(self.n_components):
                            if comp == 0:
                                if self.sparse_data:
                                    temp_scores = X[block].dot(self.W_[block][:, comp:comp+1])
                                    for sparse_row in sparse_X_info_[block][0]:
                                        non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                        temp_scores[sparse_row] = \
                                            X[block][sparse_row, non_sparse_columns].dot(
                                                self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                    T_[block] = temp_scores
                                else:
                                    T_[block] = X[block].dot(self.W_[block][:, comp:comp+1])
                            else:
                                # deflate the block
                                X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                                if self.sparse_data:
                                    temp_scores = X[block].dot(self.W_[block][:, comp:comp + 1])
                                    for sparse_row in sparse_X_info_[block][0]:
                                        non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                        temp_scores[sparse_row] = \
                                            X[block][sparse_row, non_sparse_columns].dot(
                                                self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                    T_[block] = np.hstack((T_[block], temp_scores))
                                else:
                                    T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp+1])))
                    if return_block_scores:
                        return Ts_, T_
                    else:
                        return Ts_
                else:
                    return Ts_

        else:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64, force_all_finite=not self.sparse_data)
                    if self.sparse_data:
                        sparse_X_info_[block] = self.check_sparsity_level(X[block])
            else:
                # Check dimensions
                X = check_array(X, dtype=np.float64, force_all_finite=not self.sparse_data)
                if self.sparse_data:
                    sparse_X_info_[0] = self.check_sparsity_level(X)
                X = [X]

            X_comp = np.hstack(X)
            if self.sparse_data:
                sparse_X_info_['comp'] = self.check_sparsity_level(X_comp)

            if self.sparse_data:
                temp_Ts = X_comp.dot(self.R_)
                for sparse_row in sparse_X_info_['comp'][0]:
                    non_sparse_columns = ~np.isnan(X_comp[sparse_row, :])
                    temp_Ts[sparse_row] = \
                        X_comp[sparse_row, non_sparse_columns].dot(self.R_[non_sparse_columns])
                Ts_ = temp_Ts
            else:
                Ts_ = X_comp.dot(self.R_)

            if Y is not None:
                Y = check_array(Y, dtype=np.float64, ensure_2d=False, force_all_finite=not self.sparse_data)
                if self.sparse_data:
                    sparse_Y_info_['Y'] = self.check_sparsity_level(Y)
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                    # Here the block scores are calculated iteratively for new blocks
                    T_ = []
                    for block in range(self.num_blocks_):
                        T_.append(np.empty((X[block].shape[0], 0)))
                        for comp in range(self.n_components):
                            if comp == 0:
                                if self.sparse_data:
                                    temp_scores = X[block].dot(self.W_[block][:, comp:comp+1])
                                    for sparse_row in sparse_X_info_[block][0]:
                                        non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                        temp_scores[sparse_row] = \
                                            X[block][sparse_row, non_sparse_columns].dot(
                                                self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                    T_[block] = temp_scores
                                else:
                                    T_[block] = X[block].dot(self.W_[block][:, comp:comp+1])
                            else:
                                # deflate the block
                                X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                                if self.sparse_data:
                                    temp_scores = X[block].dot(self.W_[block][:, comp:comp + 1])
                                    for sparse_row in sparse_X_info_[block][0]:
                                        non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                        temp_scores[sparse_row] = \
                                            X[block][sparse_row, non_sparse_columns].dot(
                                                self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                    T_[block] = np.hstack((T_[block], temp_scores))
                                else:
                                    T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp+1])))
                # Calculate Y scores
                if self.sparse_data:
                    temp_scores = Y.dot(self.V_)
                    for sparse_row in sparse_Y_info_['Y'][0]:
                        non_sparse_columns = ~np.isnan(Y[sparse_row, :])
                        temp_scores[sparse_row] = \
                            Y[sparse_row, non_sparse_columns].dot(self.V_[non_sparse_columns])
                    temp_scores = temp_scores / np.linalg.norm(temp_scores, axis=0)
                    U_ = temp_scores
                else:
                    U_ = Y.dot(self.V_) / np.linalg.norm(Y.dot(self.V_), axis=0)
                if return_block_scores:
                    return Ts_, T_, U_
                else:
                    return Ts_, U_
            else:
                # Here the block scores are calculated iteratively for new blocks
                T_ = []
                for block in range(self.num_blocks_):
                    T_.append(np.empty((X[block].shape[0], 0)))
                    for comp in range(self.n_components):
                        if comp == 0:
                            if self.sparse_data:
                                temp_scores = X[block].dot(self.W_[block][:, comp:comp + 1])
                                for sparse_row in sparse_X_info_[block][0]:
                                    non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                    temp_scores[sparse_row] = \
                                        X[block][sparse_row, non_sparse_columns].dot(
                                            self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                T_[block] = temp_scores
                            else:
                                T_[block] = X[block].dot(self.W_[block][:, comp:comp + 1])
                        else:
                            # deflate the block
                            X[block] = X[block] - Ts_[:, comp-1:comp].dot(self.P_[block][:, comp-1:comp].T)
                            if self.sparse_data:
                                temp_scores = X[block].dot(self.W_[block][:, comp:comp + 1])
                                for sparse_row in sparse_X_info_[block][0]:
                                    non_sparse_columns = ~np.isnan(X[block][sparse_row, :])
                                    temp_scores[sparse_row] = \
                                        X[block][sparse_row, non_sparse_columns].dot(
                                            self.W_[block][:, comp:comp + 1][non_sparse_columns])
                                T_[block] = np.hstack((T_[block], temp_scores))
                            else:
                                T_[block] = np.hstack((T_[block], X[block].dot(self.W_[block][:, comp:comp + 1])))
                if return_block_scores:
                    return Ts_, T_
                else:
                    return Ts_

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

        if self.sparse_data:
            sparse_X_info_ = {}

        if self.standardize:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64, force_all_finite=not self.sparse_data)
                    X[block] = self.x_scalers_[block].transform(X[block])
            else:
                X = check_array(X, dtype=np.float64, force_all_finite=not self.sparse_data)
                X = [self.x_scalers_[0].transform(X)]


            X = np.hstack(X)
            if self.sparse_data:
                sparse_X_info_['comp'] = self.check_sparsity_level(X)
            if self.sparse_data:
                temp_y_hat = X.dot(self.beta_)
                for sparse_row in sparse_X_info_['comp'][0]:
                    non_sparse_columns = ~np.isnan(X[sparse_row, :])
                    temp_y_hat[sparse_row] = \
                        X[sparse_row, non_sparse_columns].dot(self.beta_[non_sparse_columns])
                y_hat = self.y_scaler_.inverse_transform(temp_y_hat)
            else:
                y_hat = self.y_scaler_.inverse_transform(X.dot(self.beta_))
        else:
            if isinstance(X, list) and not isinstance(X[0], list):
                for block in range(len(X)):
                    # Check dimensions
                    X[block] = check_array(X[block], dtype=np.float64, force_all_finite=not self.sparse_data)
            else:
                X = check_array(X, dtype=np.float64, force_all_finite=not self.sparse_data)
                X = [X]
            X = np.hstack(X)
            if self.sparse_data:
                sparse_X_info_['comp'] = self.check_sparsity_level(X)
            if self.sparse_data:
                temp_y_hat = X.dot(self.beta_)
                for sparse_row in sparse_X_info_['comp'][0]:
                    non_sparse_columns = ~np.isnan(X[sparse_row, :])
                    temp_y_hat[sparse_row] = \
                        X[sparse_row, non_sparse_columns].dot(self.beta_[non_sparse_columns])
                y_hat = temp_y_hat
            else:
                y_hat = X.dot(self.beta_)

        return y_hat

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
            return metrics.explained_variance_score(Y, self.predict(X), sample_weight=None,
                                                    multioutput='variance_weighted')

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
            plt.figure()
            plt.suptitle("Component {}: {}% expl. var. in Y".format(comp+1, (100*self.explained_var_y_[comp]).round(2)), 
                         fontsize=12, fontweight='bold')

            gs1 = GridSpec(1, self.num_blocks_, top=0.875, bottom=0.85, right=0.95)
            for block in range(self.num_blocks_):
                plt.subplot(gs1[0, block])
                plt.text(0.5, 0, "X-Block {:d}\nImportance: {:.0f}%".format(block + 1, self.A_[block, comp] * 100),
                         fontsize=12, horizontalalignment='center')
                plt.axis('off')

            gs2 = GridSpec(2, self.num_blocks_, top=0.8, hspace=0.45, wspace=0.45, right=0.95)
            loading_axes = []
            score_axes = []
            # List for inverse transforming the loadings
            P_inv_trans = []
            for block in range(self.num_blocks_):
                # Inverse transforming weights/loadings
                if self.standardize:
                    P_inv_trans.append(self.x_scalers_[block].inverse_transform(self.P_[block][:, comp]))
                else:
                    P_inv_trans.append(self.P_[block][:, comp])

                if len(loading_axes) == 0:
                    # Loadings
                    loading_axes.append(plt.subplot(gs2[0, block]))
                    plt.plot(P_inv_trans[block])
                    step = int(P_inv_trans[block].shape[0] / 4)
                    plt.xticks(np.arange(0, P_inv_trans[block].shape[0], step),
                               np.arange(1, P_inv_trans[block].shape[0] + 1, step))
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
                    plt.ylabel("Block Score")
                    plt.xlabel("Sample")
                    plt.grid()
                else:
                    # Loadings
                    loading_axes.append(plt.subplot(gs2[0, block]))
                    plt.plot(P_inv_trans[block])
                    step = int(P_inv_trans[block].shape[0] / 4)
                    plt.xticks(np.arange(0, P_inv_trans[block].shape[0], step),
                               np.arange(1, P_inv_trans[block].shape[0] + 1, step))
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

        plt.figure()
        plt.suptitle("Block importances", fontsize=14, fontweight='bold')
        gs3 = GridSpec(1, 1, top=0.825, right=0.7, hspace=0.45, wspace=0.4)
        ax = plt.subplot(gs3[0, 0])
        width = 0.8 / len(num_components)
        for i, comp in enumerate(num_components):
            ax.bar(np.arange(self.num_blocks_) + 0.6 + i * width, 100 * np.ravel(self.A_[:, comp]), width=width,
                   label="Component {}".format(comp + 1))
        ax.set_xticklabels(list(np.arange(self.num_blocks_) + 1))
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.4))
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_xlabel("Block")
        ax.set_ylabel("Block importance in %")
        plt.show()
