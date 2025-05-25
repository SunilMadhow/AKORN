from online_linear_regression import LinearExpert1d
from sklearn.linear_model import LinearRegression

import numpy as np
import math
from util import data_matrix
class FLH:

	def __init__(self, X, Y, T, lr = 0.5):
		self.T = T
		self.t = 0
		self.X = X
		self.Y = Y
		self.lr = lr
		self.experts = []
		self.weights = []
		self.loss = 0
		self.predictions = []
		self.expert_errors = []

		self.W = None

	def step(self, x, y):
		self.__step(x, y)

	def __step(self, x, y):
		t = self.t

		if len(self.experts) == 0:
			pred = 0
			self.predictions.append(0)
			new_exp = LinearExpert1d(self.t + 1, [x], [y])
			self.experts.append(new_exp)
			self.weights = [1]
			self.loss = 0
			self.t = self.t + 1

		elif self.t == self.T:
			print("i ran")

		else:
			x_experts = []

			for e in self.experts:
				# print("e.history_y = ", e.history_y)
				x_experts.append(e.predict(x, t - e.start))
				e.update(y, t)

			pred = np.dot(np.array(self.weights), np.array(x_experts))
			self.predictions.append(pred)

			loss_t = (pred - y)**2
			self.loss = self.loss + loss_t

			l_experts = np.square(np.array(x_experts) - y)

			self.expert_errors.append(l_experts)

			new_weights = np.exp(-self.lr*l_experts)*np.array(self.weights)
			new_weights =  ((new_weights / np.sum(new_weights))*(1 - 1/(t + 1))).tolist()
			new_weights.append(1/(t+1))
			self.weights = new_weights

			new_exp = LinearExpert1d(self.t + 1, [x], [y])
			self.experts.append(new_exp)

			self.t = self.t + 1


	def run_fixed_weights(self, Xnew, Ynew, W):
		predictions = []
		experts = []
		for t in range(Xnew.size):
			weights = W[t][:t]
			if len(experts) == 0:
				pred = 0
				predictions.append(0)
				new_exp = LinearExpert1d(t + 1, [Xnew[t]], [Ynew[t]])
				experts.append(new_exp)
			else:
				x_experts = []
				for e in experts:
					# print("e.history_y = ", e.history_y)
					x_experts.append(e.predict(Xnew[t], t - e.start))
					e.update(Ynew[t], t)
				pred = np.dot(np.array(weights), np.array(x_experts))
				predictions.append(pred)
				new_exp = LinearExpert1d(t + 1, [Xnew[t]], [Ynew[t]])
				experts.append(new_exp)
		return predictions

		W = []
		for j in range(self.T):
			if j%200 == 0:
				print(j)
			# print(self.weights)
			w = self.weights + [0]*(self.T - j)
			
			W.append(w)
			self.__step(self.X[j], self.Y[j])
		print("loss = ", self.loss)
		W = np.array(W)
		return W

	def run(self):
		W = []
		for j in range(self.T):
			if j%1000 == 0:
				print(j)
			# print(self.weights)
			w = self.weights + [0]*(self.T - j)
			
			W.append(w)
			self.__step(self.X[j], self.Y[j])
		# print("loss = ", self.loss)
		W = np.array(W)
		self.W = W
		return W

	def get_hat_matrix(self):
		if self.W is None:
			raise NoneError("Must call run() before calling get_hat_matrix()")
		T = self.T
		X = self.X
	
		W = self.W
		H = []
		H.append([0] + [0]*(T-1))
		H.append([0,1] + [0]*(T-2))

		for i in range(2, T):
		    ai = W[i]
		    Temp = np.zeros((2,T))
		    for j in range(0, i):

		        
		        Xji = data_matrix(X[j:i]) # i - j -t0 + 1 = i - j +1 x 2
		        Bji = np.linalg.inv(Xji.T@Xji)@Xji.T #2 x i - j +1
		        Zji = np.concatenate((np.zeros((2,j)), Bji, np.zeros((2,T-i))), axis = 1)
		        Temp = Temp + ai[j]*Zji
		    Hi = np.array([1,X[i]]).T@Temp
		    H.append(Hi)

		H = np.array(H)
		return H


def addle_forward_backward(X, Y, T, var, delta = 0.1):
	std = math.sqrt(var)
	B = np.max(Y)
	lr = 1/(8*B**2)
	forw = FLH(X, Y, T, lr)
	forw.run()
	back = FLH(np.flip(X), np.flip(Y), T, lr)
	back.run()
	return (forw.predictions + np.flip(back.predictions))/2

def low_switch(X, Y, T, lr, delta, var, threshold_coef = 1, ub = True):
	op = FLH(X, Y, T, lr)
	binned_x = []
	binned_y = []

	P = T**2
	if ub == False:
		P = T

	e = LinearExpert1d()
	b = 0
	s = 0

	preds_op = []
	j = 0
	while j < T:
		# if j%1000 == 0:
		# 	print(j)
		op.step(X[j], Y[j])
		pred_al = op.predictions[-1]
		preds_op.append(pred_al)
		preds_e = [e.predict(X[i], i) for i in range(b, j + 1)]

		e.update(Y[j], j)
		s = np.sum((np.array(preds_op)[1:] - np.array(preds_e[1:]))**2)

		# if s > 0:
		if s > 5*threshold_coef*var*math.log(P/delta):
			binned_x.append(X[b:j])
			binned_y.append(Y[b:j])
			b = j
			e = LinearExpert1d()
			op = FLH(X[j + 1:], Y[j + 1: ], T, lr)
			preds_op = []
		else:
			j = j + 1

	if b != T:
	    binned_x.append(X[b:T])
	    binned_y.append(Y[b:T])
	return (binned_x, binned_y)






