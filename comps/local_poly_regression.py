import numpy as np
import math
from util import *

def boxcar(x, Xi, bandwidth):
	return abs(x - Xi)/bandwidth <= 1/2

def epanechikov(x, Xi, bandwidth):
	t = (x - Xi)/bandwidth
	if abs(t) <= 1:
		return 3*(1-t**2)/4
	else:
		return 0

def gaussian(x, Xi, bandwidth):
	return math.exp(-((x - Xi)/bandwidth)**2/2)/math.sqrt(2*math.pi)

def kernel(x, Xi, bandwidth):
	return boxcar(x, Xi, bandwidth)

def local_linear_hatmatrix_2(X, bandwidth, kernel):
	B = data_matrix(X)
	W = np.array([np.diag([kernel(x, Xi, bandwidth) for Xi in X])@B@np.linalg.inv(B.T@np.diag([kernel(x, Xi, bandwidth)for Xi in X])@B)@np.array([1,x]) for x in X])
	return W