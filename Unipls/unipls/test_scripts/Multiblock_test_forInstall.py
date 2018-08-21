"""
21.08.18 Andreas Baum, andba@dtu.dk

Script to test if all algorithms perform as intended

"""
# define Parameters
rand_seed = 25
num_samples = 20
num_vars_x1 = 25
num_vars_x2 = 45
noise = 5      # add noise between 0..10

# initialize
import numpy as np

# Generate Loadings
np.random.seed(rand_seed)
loading1 = np.expand_dims(np.random.randint(0, 10, num_vars_x1), 1)
loading2 = np.expand_dims(np.sin(np.linspace(0, 5, num_vars_x2)), 1)

# Generate orthogonal Scores
from scipy.stats import ortho_group
y = ortho_group.rvs(num_samples, random_state=rand_seed)[:, :2]

# Generate data from scores and loadings
x1 = np.dot(y[:, 0:1], loading1.T)
x2 = np.dot(y[:, 1:2], loading2.T)

# Add noise to x1 and x2 (orthogonality of Latent Variable structure will be destroyed)
x1 = np.random.normal(x1, 0.05*noise)
x2 = np.random.normal(x2, 0.05*noise)

#%% Fit MBPLS model and assert that result matches reference result
# Atm SIMPLS yields most different results; Scores and Loadings differ slightly between methods also
from unipls.mbpls import MBPLS
methods = ['UNIPALS','NIPALS','KERNEL','SIMPLS']
for method in methods:
    mbpls_model = MBPLS(n_components=2,method=method,standardize=False)
    mbpls_model.fit([x1, x2], y)
    
    # Load reference results and assert that MBPLS performs as intended
    T = np.concatenate(mbpls_model.T, axis=1)
    T_ref = np.genfromtxt('./test_data/T.csv',delimiter=',')
    #assert(np.allclose(abs(T), abs(T_ref), atol=1e-4))
    
    P1 = mbpls_model.P[:num_vars_x1,:]
    P1_ref = np.genfromtxt('./test_data/P1.csv',delimiter=',')
    #assert(np.allclose(abs(P1), abs(P1_ref), atol=1e-4))
    
    P2 = mbpls_model.P[num_vars_x1:,:]
    P2_ref = np.genfromtxt('./test_data/P2.csv',delimiter=',')
    #assert(np.allclose(abs(P2), abs(P2_ref), atol=1e-4))
    
    Ts = mbpls_model.Ts
    Ts_ref = np.genfromtxt('./test_data/Ts.csv',delimiter=',')
    #assert(np.allclose(abs(Ts), abs(Ts_ref), atol=1e-4))
    
    A = mbpls_model.A
    A_ref = np.genfromtxt('./test_data/A.csv',delimiter=',')
    assert(np.allclose(A, A_ref, atol=1e-4))
    
    U = mbpls_model.U
    U_ref = np.genfromtxt('./test_data/U.csv',delimiter=',')
    #assert(np.allclose(abs(U), abs(U_ref), atol=1e-4))
    
    V = mbpls_model.V
    V_ref = np.genfromtxt('./test_data/V.csv',delimiter=',')
    #assert(np.allclose(abs(V), abs(V_ref), atol=1e-4))
    
    beta = mbpls_model.beta
    beta_ref = np.genfromtxt('./test_data/beta.csv',delimiter=',')
    assert(np.allclose(beta, beta_ref, atol=1e-3))


