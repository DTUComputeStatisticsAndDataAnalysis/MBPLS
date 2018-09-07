"""
21.08.18 Andreas Baum, andba@dtu.dk

Script to test if all algorithms perform as intended

"""
# define Parameters
rand_seed = 25
num_samples = 50
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

# Separate Data into training and test sets
indices = np.random.choice(np.arange(num_samples),num_samples,replace=False)
train, test = indices[:round(num_samples*8/10)], indices[round(num_samples*8/10):]
x1_train, x2_train = x1[train,:], x2[train,:]
x1_test, x2_test = x1[test,:], x2[test,:]
y_train, y_test = y[train,:], y[test,:]

#%% Fit MBPLS model and assert that result matches reference result
# SIMPLS doesn't give Multiblock results, therefore left out for A (block importance) and T (block scores)
from mbpls.mbpls import MBPLS
predictions = []
methods = ['UNIPALS', 'NIPALS', 'KERNEL', 'SIMPLS']
for method in methods:
    mbpls_model = MBPLS(n_components=2,method=method,standardize=True)
    mbpls_model.fit([x1_train, x2_train], y_train)
    
    # Load reference results and assert that MBPLS performs as intended
    if method is not 'SIMPLS':
        T = np.concatenate(mbpls_model.T_, axis=1)
        T_ref = np.genfromtxt('./test_data/T_%s.csv' % method, delimiter=',')
        assert(np.allclose(abs(T), abs(T_ref)))
        
        A = mbpls_model.A_
        A_ref = np.genfromtxt('./test_data/A_%s.csv' % method, delimiter=',')
        assert(np.allclose(A, A_ref))
        
        Ts_test, T_test, U_test = mbpls_model.transform([x1_test, x2_test], y_test)
        T_test_ref = np.genfromtxt('./test_data/T_test_%s.csv' % method, delimiter=',')
        T_test = np.concatenate(T_test, axis=1)
        assert(np.allclose(abs(T_test), abs(T_test_ref)))
    
    else:
        Ts_test, U_test = mbpls_model.transform([x1_test, x2_test], y_test)
    
    Ts_test_ref = np.genfromtxt('./test_data/Ts_test_%s.csv' % method, delimiter=',')
    assert(np.allclose(Ts_test, Ts_test_ref))
    U_test_ref = np.genfromtxt('./test_data/U_test_%s.csv' % method, delimiter=',')
    assert(np.allclose(U_test, U_test_ref))
    
    P1 = mbpls_model.P_[0]
    P1_ref = np.genfromtxt('./test_data/P1_%s.csv' % method, delimiter=',')
    assert(np.allclose(abs(P1), abs(P1_ref)))
    
    P2 = mbpls_model.P_[1]
    P2_ref = np.genfromtxt('./test_data/P2_%s.csv' % method, delimiter=',')
    assert(np.allclose(abs(P2), abs(P2_ref)))
    
    Ts = mbpls_model.Ts_
    Ts_ref = np.genfromtxt('./test_data/Ts_%s.csv' % method, delimiter=',')
    assert(np.allclose(abs(Ts), abs(Ts_ref)))
    
    U = mbpls_model.U_
    U_ref = np.genfromtxt('./test_data/U_%s.csv' % method, delimiter=',')
    assert(np.allclose(abs(U), abs(U_ref)))
    
    V = mbpls_model.V_
    V_ref = np.genfromtxt('./test_data/V_%s.csv' % method, delimiter=',')
    assert(np.allclose(abs(V), abs(V_ref)))
    
    beta = mbpls_model.beta_
    beta_ref = np.genfromtxt('./test_data/beta_%s.csv' % method, delimiter=',')
    assert(np.allclose(beta, beta_ref))
    
       
    y_predict = mbpls_model.predict([x1_test, x2_test])
    y_predict_ref = np.genfromtxt('./test_data/Y_predict_test_%s.csv' % method, delimiter=',')
    assert(np.allclose(y_predict, y_predict_ref))
    
    predictions.append(y_predict)

# Assert that all methods agree in prediction  
for prediction in predictions:
    assert(np.allclose(predictions[0],prediction,atol=1e-3))

    


