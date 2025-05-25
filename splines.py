import numpy as np


class RegressionSpline:
	def __init__(self, X, Y, knots):
	    self.X = X
	    self.Y = Y
	    self.knots = knots
	    self.beta = None

	    G = []
	    for x in X:
	        G.append(self.power_basis_evals(x))
	    self.G = np.array(G)
	    self.GGi = np.linalg.pinv(self.G.T@self.G)

	def pos_part(self, z):
	    if z > 0:
	        return z
	    else:
	        return 0
	def hat_matrix(self):
	    return (self.G)@self.GGi@(self.G.T)

	def power_basis_evals(self, x):
	    a = [1,x] + [self.pos_part(x - k) for k in self.knots]
	    return a
	def train(self):
	    self.beta = np.linalg.pinv(self.G.T@self.G)@(self.G.T)@self.Y
	def train_W(self, W):
	    self.beta = np.linalg.pinv(self.G.T@np.diag(W)@self.G)@(self.G.T)@np.diag(W)@self.Y
	    

	def weighted_pred(self, x, W):
	    beta_x = np.linalg.inv(self.G.T@np.diag(W)@self.G)@(self.G.T)@np.diag(W)@self.Y
	    return np.inner(beta_x, [x, 1])
	def eva(self, x):
		if self.beta is None:
			raise NoneError("Train spline before calling eva")
		return np.inner(self.power_basis_evals(x), self.beta)

	def get_pred_variance(self, x, var): # not technically accurate, i think; conditioning on knots changes distribution of Y
	    z = self.power_basis_evals(x)
	    return np.inner(z, var*self.GGi@z)

	def get_variances(self, var):
	    return np.array([self.get_pred_variance(x, var) for x in self.X])

