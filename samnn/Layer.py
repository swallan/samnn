import numpy as np

supported_activations = 'relu'

class Layer: 
    def __init__(self, shape: int, activation: str):
        self.shape = shape
        self.activation = activation
        self.w = None
        
    def __repr__(self):
        return (
            f"Layer(w.shape={self.w.shape} ({self.activation}")
    
    def get_w(self):
        return self.w
    
    def gradient_activation(self):
        if self.activation == 'relu':
            return self.gradient_relu()
        if self.activation == 'softmax':
            # softmax - cross entropy loss derivative is already taken care of
            return 1
        if self.activation == 'sigmoid':
            return self.h_out_activate * (1 - self.h_out_activate)
        raise ValueError(f'{self.activation} not yet supported')
    
    def gradient_relu(self):
        x = self.h_out_activate
        x[~(x==0)] = 1
        return x

    def softmax(self,x):
        x = x.T
        e_x = np.exp(x - np.max(x))
        return (e_x /( e_x.sum(axis=0) + 1e-7)).T
            
    def sigmoid(self, data):
        return 1 / (1 + np.exp(-data))
    
    def activate(self, data):
        if self.activation == 'relu':
            return np.maximum(data, 0) 
        if self.activation == 'softmax':
            # The derivative of softmax is already taken
            # in the network wrt CCE.
            return self.softmax(data)
        if self.activation == 'sigmoid':
            return self.sigmoid(data)
        raise ValueError(f'{self.activation} not yet supported')
    
    
    def apply(self, data):
        '''apply the layer to the input data.'''
        if self.w is None:
            '''the shape of w is based on
            n columns - the output previous layer
            m rows    - the input shape'''
            INPUT_LAYER_SIZE = data.shape[1]
            rng = np.random.default_rng(1234)

            from numpy.random import randn
            # number of nodes in the previous layer
            n = INPUT_LAYER_SIZE# * self.shape 
            # calculate the range for the weights
            std = (2.0 / n)** .5
            # generate random numbers
            numbers = rng.standard_normal(INPUT_LAYER_SIZE * self.shape )
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
        self.h_out_activate = self.activate(self.h_out)
        return self.h_out_activate

    def gradient_w(self):
        '''
        get the gradient from this layer
        d(wx + b) / d(w) = x
        '''
        return self.h_in

    
    def update(self, gradient_chain, learning_rate=.05, last=False):
        # print(f"gradient_chain.shape:  {gradient_chain.shape}")
        # print(f"W.shape: {self.w.shape}")
        # print(f"self.h_in.shape: {self.h_in.shape}")
        '''update weights `w` for this layer'''
        # According to the chain rule, the gradient for the current layer is 
        # previous gradients * activation gradient * layer input (DW)
        # with shapes
        # `output_shape` * `output_shape` * `weights`
        # n x m       *      n x m       @     n x _
        # print(gradient_chain.shape, self.b.shape)
        gradient_current_layer = self.h_in.T @ (gradient_chain * self.gradient_activation())
        # gradient_current_layer = self.h_in.T @ (gradient_chain * self.gradient_activation())
        self.w = self.w - (gradient_current_layer * learning_rate)
        # if np.linalg.norm(self.w) > 10:
        #     self.w = (self.w / np.linalg.norm(self.w))
        self.b = self.b - (gradient_chain.sum(axis=0) * learning_rate)
        '''
        d(wx + b)/dx = w
        '''
        gradient_chain = gradient_chain @ self.w.T
        # raise ValueError(1)
        '''return gradient chain for future layer use'''
        return gradient_chain
    
    
# class ConvLayer