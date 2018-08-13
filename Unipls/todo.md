**TODO**

- [x] Add SIMPLS
- [x] Standard: Blocks have to be standardized
- [ ] Simulation of dimensions and samples (X and Y)
- [ ] Paper (Bayesian PLS): A review of variable selection methods in PLS Regression
- [x] Predict method
- [ ] Block importance, Noise (due to covariance maximization),
- [ ] What has to be done: Feature selection (MLE, Bayesian)
- [x] code cleanup
- [x] Transfer all algorithms to OOP and align structure
- [x] Plot method in MBPLS-Class
- [x] Transfer MBPLS_Lindgreen to PLS-Class

**Testing the package**

* `$ git clone https://github.com/b0nsaii/MBPLS.git`
* Install options 
    * Install as developing package
        * Move to /directory/to/MBPLS
        * `$ pip install -e pls_package`
    * Add the directory to your systempath
        * `$ export PYTHONPATH="/directory/to/MBPLS:$PYTHONPATH"
* Now you can import the MBPLS function by typing\
`from all_pls.mbpls import MBPLS`

author: Laurent Vermue, lauve@dtu.dk