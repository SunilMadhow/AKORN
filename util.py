import numpy as np
import math

def eval_model(model, point):
		return model.predict(np.array(point).reshape(-1, 1))[0]


def log_transform(H, z = 20):
	Hpos = H.clip(0)
	Hneg = H.clip(max = 0)
	Hpos = np.log(1 + z*Hpos)
	Hneg = -np.log(1 - z*Hneg)
	return Hpos + Hneg

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def second_diff(n):
	b = [0] + [2]*(n-2) + [0]
	a = [-1]*(n-2) + [0]
	c =  [0] + [-1]*(n-2)

	return tridiag(a, b, c)

def bad_seq_tv1(y):
	s = 0
	for i in range(1, len(y) - 1):
		s = s + abs(2*y[i] - y[i-1] - y[i + 1])
	return s

def seq_tv1(y, spacing):
	if len(y) == 0 or len(y) == 1:
		return 0
	if len(y) == 2:
		return abs(y[1] - y[0])
	y_ = np.array(y)
	return spacing*np.linalg.norm(second_diff(len(y))@y_, 1)

def sigmoid(x):
	return 1/(1 + math.exp(-x))

def piecewise_linear(slopes, knot_locations, x, first_slope = 0):
    assert len(slopes) == len(knot_locations)
    if x <= knot_locations[0]:
        return first_slope*x
    i = np.searchsorted(knot_locations, x) - 1
    y = np.sum([slopes[j]*(knot_locations[j + 1] - knot_locations[j]) for j in range(0, i)])
    val = y + (x - knot_locations[i])*slopes[i] + first_slope*knot_locations[0]
    return val

def count_linear_pieces(theta):
    D2 = second_diff(theta.size)
    C = np.abs(D2@theta)
    return 1 + np.count_nonzero(C[C>0.000001])

def data_matrix(X):
    return np.concatenate((np.ones((X.size, 1)), X.reshape(X.size, 1)), axis = 1)
    
def doppler(x, epsilon):
    return np.sin(2 * np.pi * (1 + epsilon) / (x + epsilon))


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def subplots_centered(nrows, ncols, figsize, nfigs):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs





