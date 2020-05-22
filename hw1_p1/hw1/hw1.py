"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0, alpha=0.9):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = [Linear(in_size, out_size, weight_init_fn, bias_init_fn)
                              for (in_size, out_size) in zip([input_size] + hiddens, hiddens + [output_size])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i], alpha) for i in range(self.bn)]

        self.input = None
        self.output = None

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        self.input = x
        self.output = x

        for k in range(len(self.linear_layers)):
            z = self.linear_layers[k].forward(self.output)
            if self.bn and k < self.bn:
                z = self.bn_layers[k].forward(z, not self.train_mode)
            self.output = self.activations[k](z)
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for layer in self.linear_layers:
            layer.zero_grads()

        if self.bn:
            for bn_layer in self.bn_layers:
                bn_layer.zero_grads()

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for layer in self.linear_layers:
            layer.step(self.lr, self.momentum)

        # Do the same for batchnorm layers
        if self.bn:
            for bn_layer in self.bn_layers:
                bn_layer.step(self.lr)

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.zero_grads()
        # loss = self.criterion.forward(self.output, labels)
        loss = self.total_loss(labels)
        dy = self.criterion.derivative()

        for k in range(self.nlayers - 1, -1, -1):
            dz = dy * self.activations[k].derivative()
            if self.bn and k < self.bn:
                dz = self.bn_layers[k].backward(dz)
            dy = self.linear_layers[k].backward(dz)

        return loss / self.input.shape[0]

    def error(self, labels):
        return (np.argmax(self.output, axis=1) != np.argmax(labels, axis=1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = _

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):
        # Per epoch setup ...
        print(f"epoch: {e + 1}")

        np.random.shuffle(idxs)

        losses = []
        errors = []

        # Train ...
        mlp.train()
        for b in range(0, len(trainx), batch_size):
            mlp.zero_grads()

            b_trainx = trainx[idxs[b: b + batch_size], :]
            b_trainy = trainy[idxs[b: b + batch_size], :]

            mlp.forward(b_trainx)
            loss = mlp.backward(b_trainy)

            error = mlp.error(b_trainy) / batch_size

            mlp.step()

            losses.append(loss)
            errors.append(error)

        training_losses[e] = np.mean(losses)
        training_errors[e] = np.mean(errors)

        losses = []
        errors = []

        # Val ...
        mlp.eval()
        for b in range(0, len(valx), batch_size):
            b_valx = valx[b: b + batch_size, :]
            b_valy = valy[b: b + batch_size, :]

            mlp.forward(b_valx)
            loss = mlp.backward(b_valy)

            error = mlp.error(b_valy) / batch_size

            losses.append(loss)
            errors.append(error)

        validation_losses[e] = np.mean(losses)
        validation_errors[e] = np.mean(errors)

    # Cleanup ...
    losses = []
    errors = []
    prediction = []

    # test...
    mlp.eval()
    for b in range(0, len(testx), batch_size):
        b_testx = testx[b: b + batch_size, :]
        b_testy = testy[b: b + batch_size, :]

        out = mlp.forward(b_testx)
        loss = mlp.backward(b_testy)
        error = mlp.error(b_testy)

        prediction.append(out.argmax(axis=1))

        losses.append(loss)
        errors.append(error)

    print("test loss: {:.6f}".format(np.mean(losses)))
    print("test errors: {:.6f}".format(np.mean(errors)))
    # Return results ...

    return training_losses, training_errors, validation_losses, validation_errors
