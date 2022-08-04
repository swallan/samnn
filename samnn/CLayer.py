import numpy as np
from numpy.lib.stride_tricks import as_strided
from Layer import Layer
from im2col import im2col_indices
import numpy as np

class CLayer(Layer):
    def apply(self, data):
        n_filters = 8
        filter_size = self.filter_size = 5

        # data comes in with shape (n x 784 x 1), need to reshape to (n, 28, 28, 1)
        h = w = int(data.shape[1] ** .5)
        d = data.reshape(data.shape[0], 1, w, h)
        self.h_in = d
        # create weight w. It will be of shape
        # (n_filters x c_channels x height x width). For now use 3, 1, 3, 3
        if self.w is None:
            # need to generate filter.
            rng = np.random.default_rng(1234)
            std = (2.0 / data.shape[1]) ** .5
            self.w = rng.standard_normal(1 * n_filters * filter_size * filter_size).reshape(1, n_filters, filter_size, filter_size) * std
            ''' np.asarray([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]])#np.ones((1, 1, 3, 3))#'''
            self.w = self.w.reshape(1, n_filters, filter_size, filter_size)
        n_data, n_channel, height, width = d.shape
        _, n_filters, f_height, f_width = self.w.shape
        stride = 1
        pad = 1

        # create storage for the output.
        out_height = int(1 + (height + 2 * pad - f_height) / stride)
        out_width = int(1 + (width + 2 * pad - f_width) / stride)

        self.x_col = im2col_indices(d, f_height, f_width, padding=1, stride=1)
        w_col = self.w.reshape(n_filters, -1)

        out = w_col @ self.x_col
        out = out.reshape(n_filters, out_height, out_width, n_data)
        out = out.transpose(3, 0, 1, 2)

        # right now the shape of the output is (n_data x n_filters x width x height). (4d)
        # reshape it to be (n_data x n_filters * width * height). (2d)
        self.h_out = out.reshape(n_data, n_filters * out_height * out_width)
        self.h_out_activate = self.activate(self.h_out)

        return self.h_out_activate

    def update(self, gradient_chain, learning_rate=.05, last=False):
        filter_size = self.filter_size
        # (x, w, b, conv_param) = cache
        # (self.data, self.w, self.b, ...)
        n_data, n_channel, height, width = self.h_in.shape
        _, n_filters, f_height, f_width = self.w.shape
        stride = 1
        pad = 1
        gradient_chain = gradient_chain.reshape(n_data, n_filters, int((self.h_out.shape[1] /n_filters) ** .5), int((self.h_out.shape[1] /n_filters) ** .5))
        gradient_chain_reshaped = gradient_chain.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        dW = gradient_chain_reshaped @ self.x_col.T
        dW = dW.reshape(1, n_filters, filter_size, filter_size)
        self.w = self.w - (dW * learning_rate)
        # w_reshape = self.w.reshape(n_filters, -1)
        # dX_col = w_reshape.T @ gradient_chain_reshaped






