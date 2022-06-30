import numpy as np

supported_activations = 'relu'

class Layer: 
    def __init__(self, shape: int, activation: str):
        self.shape = shape
        self.activation = activation
        self.w = None
    
    def get_w(self):
        return self.w
        
    def activate(self, data):
        return np.maximum(data, 0)

    def apply(self, data):
        '''apply the layer to the input data.'''
        if self.w is None:
            '''the shape of w is based on
            n columns - the output previous layer
            m rows    - the input shape'''
            INPUT_LAYER_SIZE = data.shape[1]
            self.w = np.random.rand(INPUT_LAYER_SIZE, self.shape) * np.sqrt(2.0/INPUT_LAYER_SIZE)
        self.h_in = data
        print(f"w:   :{self.w.shape}")
        self.h_out = data @ self.w

        print(f"h_o  :{self.h_out.shape}")
        return self.h_out
    
        # self.h_out_activate = self.activate(self.h_out)
        # return self.h_out_activate
    
    def gradient_w(self):
        '''get the gradient from this layer'''
        return self.h_in #* self.gradient_relu()
    
    def gradient_relu(self):
        x = self.h_out
        x[x==0] = 1
        return x
    
    def update(self, gradient_chain, learning_rate=.05, last=False):
        print('update')
        # print(f'wshape: {self.w.shape}')
        # print(f'relushape: {self.h_out_activate.shape}')

        '''update weights `w` for this layer'''
        print('gradient_chain.shae', gradient_chain.shape)

        gradient_current_layer = self.h_in.T @ gradient_chain
        print('w.shape',self.w.shape)
        print('gradient_current_layershape', gradient_current_layer.shape)
        # print('grad_curret:', gradient_current_layer)
        self.w = self.w - gradient_current_layer * learning_rate
        print('''gradient_chain = gradient_chain @ self.w''')
        print(f'''gradient_chain = {gradient_chain.shape} @ {self.w.shape}''')

        gradient_chain = gradient_chain @ self.w.T
        # print(self.h_out_activate)
        
        '''return gradient chain for future layer use'''
        return gradient_chain