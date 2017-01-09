import numpy as np


class MLP:

    def linear_gradient(self, weights, error):
        return np.dot(error, weights)

    def compute_error(self, x, w, y):
        x = x*w
        res1 = x.reshape(x.shape[0], 1)
        res = np.dot(res1, y.reshape(y.shape[0], 1).T)
        self.eta = 0.0001
        if self.Regularization == 2:
            res = res + self.eta*self.W

        if self.Regularization == 1:
            res = res + self.eta*np.sign(self.W)

        res = np.hstack((res, res1))
        return res.T

    def cost_gradients(self, weights, activation, error):
        we = self.linear_gradient(weights, error)
        ag = self.activation_gradient()
        e = [self.compute_error(a, we, b) for a,b in izip(ag, activation)]
        gW = np.mean(e, axis=0).T
        return gW

    def update_parameters(self, params):
        self.params = params
        param1 = self.params.reshape(-1, self.nparam)
        self.W = param1[:, 0:self.nparam-1]
        self.b = param1[:, self.nparam-1]



