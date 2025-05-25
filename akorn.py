from splines import RegressionSpline
from flh import FLH, low_switch
import numpy as np
import math

class AKORN:

	def __init__(self, X, Y, var, ub = False, delta = 0.1, threshold_coef = 1, lr_coeff = 1/8):
		self.X = X
		self.Y = Y
		self.var = var
		self.ub = ub
		self.delta = delta
		self.threshold_coef = threshold_coef
		self.lr_coeff = lr_coeff

		self.sp_f = None
		self.sp_b = None
		self.sp = None

		self.knots = None

		self.preds = None

	def bins_to_knots(self, binned_x): #binned_x is a list of contiguous subarrays of covariates
		N = len(binned_x)
		return [binned_x[i][-1] for i in range(N - 1)]

	# def get_intersections(self, y1, y2): #find all crossover points for y1, y2 pw linear
	# 	return self.X[np.equal(y1, y2).nonzero()]

	def get_intersections(self, y1, y2):
		if len(y1) != len(y2):
			raise ValueError("Sequences must have the same length.")
		one_bigger = (y1[0] > y2[0])
		K_intersect = []
		for i in range(0, len(y1)):
			if (y1[i] > y2[i]) != one_bigger:
				K_intersect.append(i - 1)
				one_bigger = (not one_bigger)
		return self.X[K_intersect]

	def train(self):
		X = self.X
		Y = self.Y
		var = self.var
		delta = self.delta
		threshold_coef = self.threshold_coef

		T = X.size
		ub = self.ub

		B = np.max(np.abs(Y))
		lr = self.lr_coeff/(B**2)

		std = math.sqrt(var)
		# lr = 1/(8*(1 + std*math.sqrt(math.log(2*T/delta)))**2)
		print("lr = ", lr)

		print("run")
		(binned_x_for, binned_y_for) = low_switch(X, Y, T, lr, delta, var, threshold_coef, ub)
		(binned_x_back, binned_y_back) = low_switch(np.flip(X), np.flip(Y), T, lr, delta, var, threshold_coef, ub)

		knots_for = self.bins_to_knots(binned_x_for)
		knots_back = self.bins_to_knots(binned_x_back)

		if np.in1d(np.array(knots_for),np.array(knots_back)).any():
			print("Warning: shared knot")


		sp_for = RegressionSpline(X, Y, knots_for)
		sp_back = RegressionSpline(X, Y, knots_back)
		sp_for.train()
		sp_back.train()

		sp_for_preds = np.array([sp_for.eva(x) for x in X])
		sp_back_preds = np.array([sp_back.eva(x) for x in X])

		knots_intersect = self.get_intersections(sp_for_preds, sp_back_preds)

		knots = np.unique(np.concatenate((knots_for, knots_back, knots_intersect)))
		self.knots = knots

		sp = RegressionSpline(X, Y, knots)
		sp.train()

		self.preds = np.array([sp.eva(x) for x in X]) 
		self.sp = sp
		self.sp_f = sp_for
		self.sp_b = sp_back

	def get_hat_matrix(self):
		return self.sp.hat_matrix()







