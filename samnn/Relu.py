from Layer import Layer
import numpy as np

class Relu(Layer):
    def __init__(self):
        self.lname='relu'
    def apply(self, data):
        self.h_in = data
        self.h_out = np.maximum(data, 0)
        return self.h_out

    def update(self, gradient_chain, learning_rate=.05, last=False):
        x = self.h_in.copy()
        x[x < 0] = 0
        x[x > 0] = 1
        return gradient_chain