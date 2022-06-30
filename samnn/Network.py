import numpy as np

def mse(actual, desired):
    return np.mean((actual - desired) ** 2)

class Network:
    def __init__(self, *layers):
        self.layers = layers
        
    def _forwards_propogate(self, data):
        print('_forwards_propogate')
        data_out = data
        for layer in self.layers:
            data_out = layer.apply(data_out)
            # print(data_out.shape)
        return data_out
    
    def _backwards_propgate(self, loss, learning_rate=.05):
        print('_backwards_propgate')
        loss = loss
        for layer in self.layers[::-1]:
            # print('b4',layer.get_w().shape)
            loss = layer.update(loss, learning_rate=learning_rate)
            # print('after',layer.get_w().shape)
            
    
        
    def train(self, data, target, iterations=100):
        '''run this network'''
        output = None
        prev_loss = mse(target, target)
        for i in range(iterations):
            # try:
            #     print('layer weight shapes')
            #     for l in self.layers:
            #         print(l.get_w().shape)
            # except Exception:
            #     pass
            output = self._forwards_propogate(data)
            if (output.shape != target.shape):
                raise ValueError(f'NN Output shape {output.shape} does not match truth key {target.shape}')
            loss = mse(output, target)
            # if i % 100 == 0:
            print(f'loss: {loss:.04f}\tdiff: {loss - prev_loss:.019f}')
            prev_loss = loss
            self._backwards_propgate((output - target))
        return output