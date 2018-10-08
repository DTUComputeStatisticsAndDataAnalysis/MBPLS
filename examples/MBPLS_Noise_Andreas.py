"""
08.10.18

Script to test dependency of MBPLS fit on added noise levels

author: Andreas Baum, andba@dtu.dk
"""

rand_seed = 25
num_samples = 20
num_vars_x1 = 25
num_vars_x2 = 45

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ortho_group
from scipy.spatial.distance import cosine

plt.close('all')

def analysis_with_noise(noise):
    np.random.seed(rand_seed)
    p1 = np.expand_dims(np.random.randint(0, 10, num_vars_x1), 1)
    p2 = np.expand_dims(np.sin(np.linspace(0, 5, num_vars_x2)), 1)
    
    """
    fig, ax = plt.subplots(ncols=2, figsize=(15,5))
    ax[0].plot(p1, color='blue'); ax[0].set_title('Block Loading x1 (p1)'); ax[0].set_xlabel('feature')
    ax[1].plot(p2, color='orange'); ax[1].set_title('Block Loading x2 (p2)'); ax[1].set_xlabel('feature')
    """
    
    t = ortho_group.rvs(num_samples, random_state=rand_seed)[:, 0:2]
    t1 = t[:,0:1]
    t2 = t[:,1:2]
    
    """
    plt.figure()
    plt.scatter(t1, t2)
    plt.xlabel('Score vector $t_1$', size=16)
    plt.ylabel('Score vector $t_2$', size=16)
    plt.title('The scores vectors are orthogonal ($t1^Tt2 = 0$)')
    """
    
    x1 = np.dot(t1, p1.T)
    x2 = np.dot(t2, p2.T)
    
    var_x1 = np.var(x1)
    var_x2 = np.var(x2)
    
    x1 = np.random.normal(x1, 6*noise)
    x2 = np.random.normal(x2, noise)
    var_noise_x1 = np.var(x1) - var_x1
    var_noise_x2 = np.var(x2) - var_x2
    
    """
    fig, ax = plt.subplots(ncols=2, figsize=(15,5))
    ax[0].plot(x1.T, color='blue') 
    ax[0].set_title('$X_1$ data') 
    ax[0].set_xlabel('feature', size=16)
    ax[1].plot(x2.T, color='orange') 
    ax[1].set_title('$X_2$ data') 
    ax[1].set_xlabel('feature', size=16)
    """
    
    from mbpls.mbpls import MBPLS
    mbpls_model = MBPLS(n_components=2,method='UNIPALS',standardize=False)
    mbpls_model.fit(X=[x1, x2], Y=t)
    
    
    p1_hat = mbpls_model.P_[0][:,0]
    p2_hat = mbpls_model.P_[1][:,1]
    
    """
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15,7))
    ax[0][0].plot(p1_hat, color='blue') 
    ax[0][0].set_title('Block Loading $\hat{p}_1$', size=15) 
    ax[0][0].set_xlabel('feature', size=15)
    ax[0][1].plot(p2_hat, color='orange') 
    ax[0][1].set_title('Block Loading $\hat{p}_2$', size=15) 
    ax[0][1].set_xlabel('feature', size=15)
    ax[1][0].plot(p1,color='blue',ls='--') 
    ax[1][0].set_title('ground truth: Block Loading $p_1$', size=15) 
    ax[1][0].set_xlabel('feature', size=15)
    ax[1][1].plot(p2,color='orange',ls='--') 
    ax[1][1].set_title('ground truth: Block Loading $p_2$', size=15)
    ax[1][1].set_xlabel('feature', size=15)
    plt.tight_layout()
    """
    
    #t1_hat = mbpls_model.T_[0][:,0]
    #t2_hat = mbpls_model.T_[1][:,1]
    t1_hat = mbpls_model.Ts_[:,0]
    t2_hat = mbpls_model.Ts_[:,1]
    
    """
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    ax[0].scatter(t1, t1_hat) 
    ax[0].set_title('Block Scores $\hat{t}_1$ vs. ground truth $t_1$')
    ax[0].set_xlabel('$t_1$', size=15)
    ax[0].set_ylabel('$\hat{t}_1$', size=15)
    ax[1].scatter(t2, t2_hat)
    ax[1].set_title('Block Scores $\hat{t}_2$ vs. ground truth $t_2$')
    ax[1].set_xlabel('$t_2$', size=15)
    ax[1].set_ylabel('$\hat{t}_2$', size=15)
    """
    
    variances_x = mbpls_model.explained_var_xblocks_
    blockimportances = mbpls_model.A_
    variance_y = mbpls_model.explained_var_y_
    
    import pandas as pd
    variances_x = pd.DataFrame(data=variances_x.T, columns=['expl. var. X1', 'expl. var. X2'], index=['LV1', 'LV2'])
    variance_y = pd.DataFrame(data=variance_y, columns=['expl. var. Y'], index=['LV1', 'LV2'])
    blockimportances = pd.DataFrame(data=blockimportances.T, columns=['block importance X1', 'block importance X2'], index=['LV1', 'LV2'])
    pd.concat((variances_x, blockimportances, variance_y), axis=1).round(3)
    
    A = mbpls_model.A_
    SNx1 = var_x1 / var_noise_x1
    SNx2 = var_x2 / var_noise_x2
    
    return A, t1, t1_hat, t2, t2_hat, p1, p1_hat, p2, p2_hat, SNx1, SNx2

A = []
T1 = []
T1_hat = []
T2 = []
T2_hat = []
P1 = [] 
P1_hat = [] 
P2 = []
P2_hat = [] 
SNx1 = [] 
SNx2 = []

for noise in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    result = analysis_with_noise(noise)
    A.append(result[0])
    T1.append(result[1])
    T1_hat.append(result[2])
    T2.append(result[3])
    T2_hat.append(result[4])
    P1.append(result[5])
    P1_hat.append(result[6])
    P2.append(result[7])
    P2_hat.append(result[8])
    SNx1.append(result[9])
    SNx2.append(result[10])
    
# Plot BIPs in dependency of added noise
fig, ax = plt.subplots(ncols=2)
ax[0].scatter(SNx1,[a[0,0] for a in A])
ax[0].set_xlabel('S/N ratio')
ax[0].set_ylabel('Block importance')
ax[0].set_title('X1')
ax[1].scatter(SNx2,[a[1,1] for a in A])
ax[1].set_xlabel('S/N ratio')
ax[1].set_ylabel('Block importance')
ax[1].set_title('X2')
plt.tight_layout()

# Plot correlation coefficient R² of block scores in dependency of added noise
fig, ax = plt.subplots(ncols=2)
ax[0].scatter(SNx1,[abs(np.corrcoef(a[:,0], b)[0,1]) for a, b in zip(T1, T1_hat)])
ax[0].set_xlabel('S/N ratio')
ax[0].set_ylabel('correlation')
ax[0].set_title('X1')
ax[1].scatter(SNx2,[abs(np.corrcoef(a[:,0], b)[0,1]) for a, b in zip(T2, T2_hat)])
ax[1].set_xlabel('S/N ratio')
ax[1].set_ylabel('correlation')
ax[1].set_title('X2')
plt.tight_layout()

# Plot Cosine distances in dependency of added noise
fig, ax = plt.subplots(ncols=2)
ax[0].scatter(SNx1, [abs(1 - cosine(a, b)) for a, b in zip(P1, P1_hat)])
ax[0].set_xlabel('S/N ratio')
ax[0].set_ylabel('cosinus distance p1 versus p1_hat')
ax[0].set_title('X1')
ax[1].scatter(SNx2, [abs(1 - cosine(a, b)) for a, b in zip(P2, P2_hat)])
ax[1].set_xlabel('S/N ratio')
ax[1].set_ylabel('cosinus distance p2 versus p2_hat')
ax[1].set_title('X2')
plt.tight_layout()

    

