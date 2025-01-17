import numpy as np

class OnlineLinearRegression:
    """
    Online linear regression in O(1) mem & compute
    """
    def __init__(self):
        # Average of all x seen
        self.x_avg = 0.
        # Average of all y seen
        self.y_avg = 0.
        # Covariance of all x and y seen
        self.xy_covar = 0.
        # Variance of all x seen
        self.x_var = 0.
        # Number of observations seen
        self.n = 0

        self.last_y = 0

    def parameters(self):
        """
        :return: the parameters of the linear regression (beta, alpha) such that y = beta * x + alpha. If there are
        less than 2 observations, returns (None, None).
        """
        if self.n < 2:
            return 0, 0
        else:
            beta = self.xy_covar / self.x_var
            alpha = self.y_avg - beta * self.x_avg
            return beta, alpha

    def predict(self, x):
        if self.n == 0:
            return 0
        elif self.n == 1:
            return self.last_y
        b, a = self.parameters()
        return b*x + a

    def update_multiple(self, new_x: np.ndarray, new_y: np.ndarray):
        assert len(new_x) == len(new_y)

        self.last_y = new_y[-1]

        new_n = self.n + len(new_x)

        new_x_avg = (self.x_avg * self.n + np.sum(new_x)) / new_n
        new_y_avg = (self.y_avg * self.n + np.sum(new_y)) / new_n

        if self.n:
            x_star = (self.x_avg * np.sqrt(self.n) + new_x_avg * np.sqrt(new_n)) / (np.sqrt(self.n) + np.sqrt(new_n))
            y_star = (self.y_avg * np.sqrt(self.n) + new_y_avg * np.sqrt(new_n)) / (np.sqrt(self.n) + np.sqrt(new_n))
        else:
            x_star = new_x_avg
            y_star = new_y_avg

        self.n = new_n
        self.x_avg = new_x_avg
        self.y_avg = new_y_avg

        self.x_var = self.x_var + np.sum((new_x - x_star) ** 2)
        self.xy_covar = self.xy_covar + np.sum((new_x - x_star).reshape(-1) * (new_y - y_star).reshape(-1))

    def update(self, x: float, y: float):
        self.update_multiple(np.array([x]), np.array([y]))

class LinearExpert1d: # when loss function is specified by a response variable, perform linear regression

    def __init__(self, initial_time = 0, history_x_arg = None, history_y_arg = None):
        self.start = initial_time
        self.predictions = []
        

        if (history_x_arg == None) and (history_y_arg == None):
            self.last_x = 0
            self.learner = OnlineLinearRegression()

        else:
            self.last_x = history_x_arg[-1]
            self.learner = OnlineLinearRegression()
            self.learner.update_multiple(history_x_arg, history_y_arg)


    def __str__(self):
        return "started at time " + str(self.start)

    def eval_model(self, model, point):
        return model.predict(np.array(point).reshape(-1, 1))[0]

    def predict(self, x, t):
        self.last_x = x
        return self.learner.predict(x)

    def update(self, response, t):
        self.learner.update(self.last_x, response)