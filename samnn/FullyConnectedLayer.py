from Layer import Layer
import numpy as np

class FullyConnectedLayer(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.w = None
        self.lname = f'fc{shape}'

    def apply(self, data):
        if self.w is None:
            '''the shape of w is based on
            n columns - the output previous layer
            m rows    - the input shape'''
            INPUT_LAYER_SIZE = data.shape[1]
            rng = np.random.default_rng(1234)

            from numpy.random import randn
            # number of nodes in the previous layer
            n = INPUT_LAYER_SIZE  # * self.shape
            # calculate the range for the weights
            std = (2.0 / n) ** .5
            # generate random numbers
            numbers = rng.standard_normal(INPUT_LAYER_SIZE * self.shape)
            # scale to the desired range
            scaled = numbers * std
            self.w = scaled.reshape(INPUT_LAYER_SIZE, self.shape)
            n = self.shape
            # calculate the range for the weights
            std = (2.0 / n) ** .5
            # generate random numbers
            numbers = rng.standard_normal(self.shape)
            # scale to the desired range
            scaled = numbers * std
            self.b = scaled.reshape(1, self.shape)
            # self.w = rng.normal(loc=0, scale=np.sqrt(2.0/INPUT_LAYER_SIZE), size=(INPUT_LAYER_SIZE, self.shape))
            # self.b = rng.normal(loc=0, scale=np.sqrt(2.0/INPUT_LAYER_SIZE), size=(1, self.shape))
        self.h_in = data
        self.h_out = (data @ self.w) + self.b
        # self.h_out_activate = self.activate(self.h_out)
        return self.h_out

    def update(self, gradient_chain, learning_rate=.05, last=False):
        gradient_current_layer = self.h_in.T @ gradient_chain
        self.w = self.w - (gradient_current_layer * learning_rate)
        self.b = self.b - (gradient_chain.sum(axis=0) * learning_rate)
        '''
        d(wx + b)/dx = w
        '''
        gradient_chain = gradient_chain @ self.w.T
        return gradient_chain# / gradient_chain.max()

    def has_w(self):
        return True