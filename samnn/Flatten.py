from Layer import Layer

class Flatten(Layer):
    def __init__(self, b):
        self.b = b
        self.lname='flat'
    def apply(self, data):
        self.h_in = data
        self.h_out = data.reshape(data.shape[0], -1)
        if self.b:
            self.h_out = self.h_out.reshape(self.h_out.shape[0], self.h_out.shape[1])
        return self.h_out


    def update(self, gradient_chain, learning_rate=.05, last=False):
        return gradient_chain.reshape(self.h_in.shape)
