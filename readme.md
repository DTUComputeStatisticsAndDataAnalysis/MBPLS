13.03.18

- NIPALS, SIMPLS and UNIPALS PLS algorithms versus Scikit Learn PLS result (in PLS folder)
- MBPLS implementation as described in Bougeard et al 2011 (MBPLS.py)
- MBPLS.py: MBPLS results are similar to ade4 mbpls.r results
- MBPLS_Lindgreen.py: similar results as in MBPLS.py, but superscores are normalized to length 1. Regression vector beta is calculated (similar to ScikitLearn PLS result). When N > P algorithm based on SVD(X'YY'X), when N < P algorithm based on SVD(XX'YY'). This decreases computational resources a lot when accounting for very high number of features. 
- MBPLS works for univariate and multivariate Y
- MBPLS/DataSimulationExamples/ contains several simple datasets to illustrate meaningfulness of MBPLS

TODO

- code cleanup
- optional block weighting 

author: Andreas Baum, andba@dtu.dk 
