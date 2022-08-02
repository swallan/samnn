import numpy as np
from numpy.lib.stride_tricks import as_strided
from Layer import Layer

import numpy as np

class CLayer(Layer):
    def apply(self, data):
        # data comes in with shape (n x 784 x 1), need to reshape to (n, 28, 28, 1)
        d = data.reshape(data.shape[0], 1, 28, 28)

        # create weight w. It will be of shape
        # (n_filters x c_channels x height x width). For now use 3, 1, 3, 3
        if self.w is None:
            # need to generate filter.
            rng = np.random.default_rng(1234)
            std = (2.0 / data.shape[1]) ** .5
            self.w = rng.standard_normal(1 * 1 * 3 * 3).reshape(1, 1, 3, 3) * std
            ''' np.asarray([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]])#np.ones((1, 1, 3, 3))#'''
            self.w = self.w.reshape(1, 1, 3, 3)
        n_data, n_channel, height, width = d.shape
        n_filters, _, f_height, f_width = self.w.shape
        stride = 1
        pad = 1

        # create storage for the output.
        out_height = 1 + (height + 2 * pad - f_height) / stride
        out_width = 1 + (width + 2 * pad - f_width) / stride
        out = np.zeros((n_data, n_filters, int(out_height), int(out_width)))

        for n in range(n_data):
            data_padded = np.pad(d[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
            for f in range(n_filters):
                for h in range(int(out_height)):
                    for w in range(int(out_width)):
                        h1 = int(h * stride)
                        h2 = int(h * stride + f_height)
                        w1 = int(w * stride)
                        w2 = int(w * stride + f_width)
                        window = data_padded[:, h1:h2, w1:w2]
                        out[n, f, h, w] = np.sum(window * self.w[f, :, :, :])
        self.h_in = d
        self.h_out = out
        self.h_out_activate = self.activate(self.h_out)
        return self.h_out_activate.reshape(n_data, 28 * 28)

    def update(self, gradient_chain, learning_rate=.05, last=False):
        # (x, w, b, conv_param) = cache
        # (self.data, self.w, self.b, ...)
        n_data, n_channel, height, width = self.h_in.shape
        n_filters, _, f_height, f_width = self.w.shape
        stride = 1
        pad = 1


        gradient_chain = gradient_chain.reshape(n_data, 1, 28, 28)
        _, _, out_height, out_width = gradient_chain.shape

        dx = np.zeros_like(self.h_in)
        dw = np.zeros_like(self.w)

        for n in range(n_data):
            dx_pad = np.pad(dx[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
            x_pad = np.pad(self.h_in[n, :, :, :], ((0, 0), (pad, pad), (pad, pad)),
                           'constant')
            for f in range(n_filters):
                for h in range(int(out_height)):
                    for w in range(int(out_width)):
                        h1 = h * stride
                        h2 = h * stride + f_height
                        w1 = w * stride
                        w2 = w * stride + f_width
                        dx_pad[:, h1:h2, w1:w2] += self.w[f, :, :, :] * gradient_chain[
                            n, f, f_height, f_width]
                        dw[f, :, :, :] += x_pad[:, h1:h2, w1:w2] * gradient_chain[
                            n, f, f_height, f_width]
            dx[n, :, :, :] = dx_pad[:, 1:-1, 1:-1]

