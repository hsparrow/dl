# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def zero_grads(self):
        self.dbeta = np.zeros(self.beta.shape)
        self.dgamma = np.zeros(self.gamma.shape)

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        self.x = x

        if eval:
            self.norm = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            self.mean = np.mean(self.x, axis=0)
            self.var = np.var(self.x, axis=0)
            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)

            self.running_mean = self.alpha * self.running_mean + (1. - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1. - self.alpha) * self.var

        self.out = self.gamma * self.norm + self.beta

        return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        dx_norm = delta * self.gamma

        d_var = np.sum(dx_norm * (self.x - self.mean) * (-1 / 2) *
                       ((self.var + self.eps) ** (-3 / 2)), axis=0)

        d_mu = np.sum(dx_norm * (-1) * ((self.var + self.eps) ** (-1 / 2)), axis=0)

        d_x = dx_norm * ((self.var + self.eps) ** (-1 / 2)) + \
              d_var * 2 * (self.x - self.mean) / self.x.shape[0] + d_mu / self.x.shape[0]

        self.dbeta = np.sum(delta, axis=0).reshape(1, -1)
        self.dgamma = np.sum(delta * self.norm, axis=0).reshape(1, -1)

        return d_x
