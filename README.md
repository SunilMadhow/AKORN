In order to attain optimal rates, state-of-the-art algorithms for non-parametric
2 regression require that a hyperparameter be tuned according to the smoothness
3 of the ground truth [Tibshirani, 2014]. This amounts to an assumption of oracle
4 access to certain features of the data-generating process. We present a parameter-
5 free algorithm for offline non-parametric regression over T V1 -bounded functions.
6 By feeding offline data into an optimal online denoising algorithm styled after
7 Baby et al. [2021], we are able to use change-points to adaptively select knots
8 that respect the geometry of the underlying ground truth. We call this procedure
9 AKORN (Adaptive Knots generated Online for Regression spliNes). By combining
10 forward and backward passes over the data, we obtain an estimator whose empirical
11 performance is close to Trend Filtering [Kim et al., 2009, Tibshirani, 2014], even
12 when we provide the latter with oracle knowledge of the ground truth’s smoothness.
