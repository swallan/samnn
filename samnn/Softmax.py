from Layer import Layer
import numpy as np

class Softmax(Layer):
    def __init__(self):
        self.lname = 'softmax'
    def apply(self, data):
        data = data.T
        e_x = np.exp(data - np.max(data))
        return (e_x /(e_x.sum(axis=0) + 1e-7)).T

    def update(self, gradient_chain, learning_rate=.05, last=False):
        return gradient_chain + .00001