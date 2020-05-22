# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)

        self.input = None

    def zero_grads(self):
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def step(self, lr, momentum=0.0):
        if momentum == 0.0:
            self.W -= lr * self.dW
            self.b -= lr * self.db
        else:
            self.momentum_W = momentum * self.momentum_W - lr * self.dW
            self.momentum_b = momentum * self.momentum_b - lr * self.db
            self.W += self.momentum_W
            self.b += self.momentum_b

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.input = x
        return np.matmul(x, self.W) + self.b

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        self.dW = (np.matmul(self.input.T, delta) / self.input.shape[0]).reshape(self.W.shape)
        self.db = (np.sum(delta, axis=0) / self.input.shape[0]).reshape(self.b.shape)
        return np.matmul(delta, self.W.T)
