import numpy as np
from collections import namedtuple
def mse(actual, desired):
    return np.mean((actual - desired) ** 2)

def cce(actual, targets):
    return -np.sum(targets * np.log(actual + 1e-7))

def dMse(actual, targets):
    return 2 * (targets - actual)

def dCce(actual, targets):
    return actual - targets

class Network:
    losses = [np.nan]
    def __init__(self, *layers):
        self.layers = layers
        if layers[-1].activation == 'softmax':
            self.loss = cce
            self.dLoss = dCce
        else:
            self.loss = mse
            self.dLoss = dMse
        
    def _forwards_propogate(self, data):
        data_out = data
        for layer in self.layers:
            data_out = layer.apply(data_out)
            # data_out = data_out/data_out.max()
        return data_out
    
    def _backwards_propgate(self, loss, learning_rate=.05):
        chain_gradient = loss
        for layer in self.layers[::-1]:
            # print(f'prop {layer}')
            chain_gradient = layer.update(chain_gradient,
                                          learning_rate=learning_rate)

    def trainable_param(self):
        return sum([l.w.shape[0] * l.w.shape[1] for l in self.layers])
        
    def train(self, data, target, epochs=100, iterations=100, lr=.001, rng=np.random.default_rng(1234)):
        output = None

        for i in range(epochs):
            # for each epoch, get a random subsample
            subsample = rng.integers(low=0, high=len(data), size=200)
            current_data = data[subsample]
            current_target = target[subsample]
            for j in range(iterations):

                output = self._forwards_propogate(current_data)

                # The last layer should have the same dimension as the truth key
                if (output.shape[1] != target.shape[1]):
                    raise ValueError(f'NN Output shape {output.shape[1]} does not match truth key {target.shape[1]}')

                # get the loss for this output
                loss = self.loss(output, current_target)



                # _backwards_prop takes in the derivative of the loss function.
                # This is CCE.
                dLoss = self.dLoss(output, current_target)
                # print('dloss', dLoss)
                self._backwards_propgate(dLoss, learning_rate=lr)
                if np.any(np.isnan(output)):
                    print(output)
                    raise ValueError(1)
            self.losses.append(loss)
            print(f'epoch:{i} loss: {loss:.04f}\tdiff: {loss - self.losses[-2]:.019f}')

        return loss, self._forwards_propogate(data)
    
    def eval(self, data):
        return self._forwards_propogate(data)
        