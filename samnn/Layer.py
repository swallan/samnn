import numpy as np

supported_activations = 'relu'

class Layer:
    '''
    all layers must have attributes:
    - in  : the data the layer took in
    - out : the data the layer outputs
    and methods:
    - apply(input data): -> output data
    - update(downstream gradients): -> updated gradient chain
    '''
    # def __init__(self, shape: int, activation: str):
    #     self.shape = shape
    #     self.activation = activation
    #     self.w = None
    #
    # def __repr__(self):
    #     return (
    #         f"Layer(w.shape={self.w.shape}")
    
    # def get_w(self):
    #     return self.w
    #
    # def gradient_activation(self):
    #     if self.activation == 'relu':
    #         return self.gradient_relu()
    #     if self.activation == 'softmax':
    #         # softmax - cross entropy loss derivative is already taken care of
    #         return 1
    #     if self.activation == 'sigmoid':
    #         return self.h_out_activate * (1 - self.h_out_activate)
    #     raise ValueError(f'{self.activation} not yet supported')
    #
    # def gradient_relu(self):
    #     x = self.h_out_activate
    #     x[~(x==0)] = 1
    #     return x

    # def softmax(self,x):
    #     x = x.T
    #     e_x = np.exp(x - np.max(x))
    #     return (e_x /( e_x.sum(axis=0) + 1e-7)).T
    #
    # def sigmoid(self, data):
    #     return 1 / (1 + np.exp(-data))
    #
    # def activate(self, data):
    #     if self.activation == 'relu':
    #         return np.maximum(data, 0)
    #     if self.activation == 'softmax':
    #         # The derivative of softmax is already taken
    #         # in the network wrt CCE.
    #         return self.softmax(data)
    #     if self.activation == 'sigmoid':
    #         return self.sigmoid(data)
    #     raise ValueError(f'{self.activation} not yet supported')

    def apply(self, data):
        """ Feed data through the layer.

        :param data: array of proper shape input
        :return: result of data after layer applied
        """
        return data
    
    def update(self, gradient_chain, learning_rate=.05, last=False):
        """Update layer weights and pass along gradient chain.

        :param gradient_chain: downstream gradient chan
        :param learning_rate: how fast to update gradients
        :return: the updated gradient chain for upstream use.
        """
        return gradient_chain

    def has_w(self):
        return False
