- NIPALS, SIMPLS and UNIPALS PLS algorithms versus Scikit Learn PLS result
- MBPLS implementation as described in Bougeard et al 2011
- MBPLS works for univariate and multivariate Y
- MBPLS results are similar to ade4 mbpls.r results


TODO

- MBPLS algorithm needs to be optimized for the case of p >> n (using SVD(XX'YY') as described in Lindgreen et al 1998) 
- export loadings P to calculate regression vector b

author: Andreas Baum, andba@dtu.dk 
