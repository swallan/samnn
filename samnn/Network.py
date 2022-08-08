import numpy as np
from collections import namedtuple

from scipy import ndimage


def mse(actual, desired):
    return np.mean((actual - desired) ** 2)

def cce(actual, targets):
    return -np.sum(targets * np.log(actual + 1e-7))/actual.shape[0]

def dMse(actual, targets):
    return 2 * (targets - actual)

def dCce(actual, targets):
    return actual - targets

class Network:
    losses = [np.nan]
    def __init__(self, *layers, **kwds):
        self.layers = layers
        self.loss = cce
        self.dLoss = dCce
        if kwds.get('names') is not None:
            self.names = kwds.get('names')

    def _forwards_propogate(self, data):
        data_out = data
        for layer in self.layers:
            # print(data_out.shape)
            data_out = layer.apply(data_out)
        return data_out
    
    def _backwards_propgate(self, loss, learning_rate=.05):
        chain_gradient = loss
        for layer in self.layers[::-1]:
            chain_gradient = layer.update(chain_gradient,
                                          learning_rate=learning_rate)

    def trainable_param(self):
        return sum([l.w.shape[0] * l.w.shape[1] for l in self.layers])
        
    def train(self, data, target, epochs=100, iterations=100, lr=.001,
              rng=np.random.default_rng(1234), **kwds):
        output = None
        diffs = []
        import time

        for i in range(epochs):
            t0 = time.time()
            # for each epoch, get a random subsample
            subsample = rng.integers(low=0, high=len(data), size=100)
            current_data = data[subsample]
            current_target = target[subsample]
            op = self._forwards_propogate(data)
            l = self.loss(op, target)
            self.losses.append(l)
            if l - self.losses[-2] > 0:
                lr = lr / 2

            if np.mean(np.abs(diffs[-3:])) < .01:
                break
            diffs.append(l - self.losses[-2])
            output_label = np.argmax(op, axis=1)
            print(
                f"train: {np.count_nonzero(output_label == np.argmax(target, axis=1)) / len(target) * 100:.02f}% correct prediction")
            if 'val_input' in kwds.keys() :
                validation_set = kwds.get('val_input')
                val_output = self._forwards_propogate(validation_set)
                val_output = np.argmax(val_output, axis=1)

                print(f"val:   {np.count_nonzero(val_output == np.argmax(kwds.get('val_target'), axis=1)) / len(target) * 100:.02f}% correct prediction")

            print(f'epoch:{i} loss: {l:.04f}\tdiff: {l - self.losses[-2]:.03f} lr{lr}', end='')
            self.visualize(data, target)

            # for layer in self.layers:
            #     print(layer)
            #     if layer.has_w():
            #         print(f"max: {layer.w.max():.02f}, min: {layer.w.min():.02f}, mean: {layer.w.mean():.02f}")
            for j in range(iterations):
                # print(f"iter: {j}")

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
            print(f" | {time.time() - t0:000.02f} s for this epoch")



        return loss, self._forwards_propogate(data)
    
    def eval(self, data):
        return self._forwards_propogate(data)

    def visualize(self, data, target):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(len(self.layers), 10))
        nd = np.random.randint(low=0, high=len(data), size=1)

        d = data[int(nd)][np.newaxis, :, :,]
        data_out = d
        plt.subplot(10, 10, 1)
        plt.imshow(ndimage.rotate(d.reshape(d.shape[1], d.shape[2], 3), -90))
        plt.axis('off')
        plt.title('input image')
        for li, layer in enumerate(self.layers):
            data_out = layer.apply(data_out)
            if layer.lname == 'fc10':
                print(data_out)

            if layer.lname == 'softmax':
                print(data_out)
            for li_f in range(data_out.shape[-1]):
                if li_f < 10:

                    try:
                        plt.subplot(10, 10, (li + 1) * 10 + li_f +1)  # cols, rows
                        plt.axis('off')

                        if layer.lname == 'softmax':
                            plt.title(self.names[li_f])
                        else:
                            plt.title(layer.lname[:-1] + f'{li_f}')# + f'{data_out[0][li_f]:.02f}')
                        plt.set_cmap('binary')
                        if layer.lname.startswith('f') or layer.lname == ('softmax') or len(data_out.shape) == 2:
                            plt.imshow([[data_out[0][li_f]]], vmin=data_out.min(), vmax=data_out.max() )
                            # plt.ylabel(data_out[0][li_f])
                        else:
                            _, h, w = data_out[:, :, :, li_f].shape
                            plt.imshow(ndimage.rotate(data_out[:, :, :, li_f].reshape(h, w), -90))
                    except Exception as e:
                        print(e)
        label = self.names[np.argmax(target[nd])]
        plt.suptitle(f"Visualization for image of `{label}`\nPrediction: `{self.names[np.argmax(data_out)]}`")

        plt.show()

