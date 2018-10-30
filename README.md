# Partial Least Squares Package
An easy to use Python package for (Multiblock) Partial Least Squares prediction modelling of univariate or multivariate outcomes. Four state of the art algorithms have been implemented and optimized for robust performance on large data matrices. The package has been designed to be able to handle missing data, such that application is straight forward using the commonly known Scikit-Learn syntax and its model selection toolbox.   

## Installation

* Install the package for Python3 using the following command. Some dependencies might require an upgrade (scikit-learn, numpy and scipy). 
`$ pip install mbpls`
* Now you can import the MBPLS class by typing\
`from mbpls.mbpls import MBPLS`

## Quick Start

### Use the mbpls package for Partial Least Squares (PLS) prediction modeling
```python
import numpy as np
from mbpls.mbpls import MBPLS

num_samples = 40
num_features = 200

# Generate random data matrix X
x = np.random.rand(num_samples, num_features)

# Generate random reference vector y
y = np.random.rand(num_samples,1)

# Establish prediction model using 2 latent variables (components)
pls = MBPLS(n_components=2)
pls.fit(x,y)
y_pred = pls.predict(x)
```

### The mbpls package for Multiblock Partial Least Sqaures (MB-PLS) prediction modeling
```python
import numpy as np
from mbpls.mbpls import MBPLS

num_samples = 40
num_features_x1 = 200
num_features_x2 = 250

mbpls = MBPLS(n_components=3)
x1 = np.random.rand(num_samples, num_features_x1)
x2 = np.random.rand(num_samples, num_features_x2)

y = np.random.rand(num_samples, 1)

mbpls.fit([x1, x2],y)
mbpls.plot(num_components=3)

y_pred = mbpls.predict([x1, x2])
```

More elaborate (real-world) examples can be found at https://github.com/b0nsaii/MBPLS/tree/Package_OOP/examples

