# define Parameters
rand_seed = 25
num_samples = 65
num_vars_x1 = 25
num_vars_x2 = 45
noise = 70      # add noise between 0..10

# initialize
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
sb.set_style(style="whitegrid")
plt.close('all')

# Generate Loadings
np.random.seed(rand_seed)
loading1 = np.expand_dims(np.random.randint(0,10,num_vars_x1),1)
loading2 = np.expand_dims(np.sin(np.linspace(0,5,num_vars_x2)),1)

# Generate orthogonal Scores
from scipy.stats import ortho_group
y = ortho_group.rvs(num_samples, random_state=rand_seed)[:,:2]

# Generate data from scores and loadings
x1 = np.dot(y[:,0:1],loading1.T)
x2 = np.dot(y[:,1:2],loading2.T)

# Add noise to x1 and x2 (orthogonality of Latent Variable structure will be destroyed)
x1 = np.random.normal(x1,0.05*noise)
x2 = np.random.normal(x2,0.05*noise)


#%% Standardize x blocks
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()
#x1 = scaler1.fit_transform(x1)         # Scalers should be used to illustrate importance, but inverse transform missing atm
#x2 = scaler2.fit_transform(x2)

#%% Fit MBPLS model
from unipls.mbpls import MBPLS
mbpls_model = MBPLS(n_components=2)
mbpls_model.fit([x1, x2], y)


#%% Plot loadings, scores and MBPLS results
# Plot Loadings
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(loading1,color='orange'); ax[0].set_title('Loading x1'); ax[0].set_xlabel('feature number'); ax[0].set_ylabel('value')
ax[1].plot(loading2,color='blue'); ax[1].set_title('Loading x2'); ax[1].set_xlabel('feature number'); ax[1].set_ylabel('value')

# Scatter y1 versus y2
plt.figure()
plt.scatter(y[:,0:1],y[:,1:2])
plt.xlabel('y1'); plt.ylabel('y2'); plt.title('vectors y1 and y2 are orthogonal')

# Plot x1 and x2 data
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(x1.T,color='orange'); ax[0].set_title('data x1'); ax[0].set_xlabel('feature number'); ax[0].set_ylabel('value')
ax[1].plot(x2.T,color='blue'); ax[1].set_title('data x2'); ax[1].set_xlabel('feature number'); ax[1].set_ylabel('value')

# Plot MBPLS result
mbpls_model.plot([1])
mbpls_model.plot([2])

# Scatter BLock scores versus y
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(y[:,0],mbpls_model.T[0][:,0]); ax[0].set_xlabel('y1'); ax[0].set_ylabel('Block Scores x1 LV1')
ax[1].scatter(y[:,1],mbpls_model.T[1][:,1]); ax[1].set_xlabel('y2'); ax[1].set_ylabel('Block Scores x2 LV2')
plt.show()