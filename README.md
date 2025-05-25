### AKORN

In order to attain optimal rates, state-of-the-art algorithms for non-parametric
regression require that a hyperparameter be tuned according to the smoothness
of the ground truth [Tibshirani, 2014]. This amounts to an assumption of oracle
access to certain features of the data-generating process. We present a parameter-
free algorithm for offline non-parametric regression over T V1 -bounded functions.
By feeding offline data into an optimal online denoising algorithm styled after
Baby et al. [2021], we are able to use change-points to adaptively select knots
that respect the geometry of the underlying ground truth. We call this procedure
AKORN (Adaptive Knots generated Online for Regression spliNes). By combining
forward and backward passes over the data, we obtain an estimator whose empirical
performance is close to Trend Filtering [Kim et al., 2009, Tibshirani, 2014], even
when we provide the latter with oracle knowledge of the ground truth’s smoothness.
