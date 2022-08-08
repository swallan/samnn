import numpy as np
from numpy.lib.stride_tricks import as_strided
# from Layer import Layer
# from im2col import conv_im2col, conv_im2col_backprop
import numpy as np

class CLayer():
    def __init__(self, **kwds):

        rng = np.random.default_rng(1234)
        std = (2.0 / 9) ** .5
        k, k, n_channels, n_filters = 5, 5, kwds['n_channels'], kwds['n_filters']
        self.n_filters = n_filters
        self.lname = f'conv{n_filters}'
        self.w = (rng.standard_normal(k * k * n_channels * n_filters)
                  .reshape(k, k, n_channels, n_filters) * std)
        self.stride = kwds['stride']
        self.padding = 'same' # output is the same size as input
        # self.n_filters = 8
        self.k = 5

    # def calculate_output_dimensions(self, input):
    #     # calculate padding needed on height and width


    def apply(data):
        """

        :param data: input with shape (n_data, data_height, data_width, n_channels)
        :output result with shape (n_data, out_height, out_width, n_filters)
        """

        self.h_in = data
        n_data, data_height, data_width, n_channels = self.h_in.shape

        k, k, _, n_filters = self.w.shape

        # need to pad the data input on each side to make it so the sliding
        # window does not go out of bounds.
        # pad only height x width, not n_data or nchannel
        p = (k - 1) // 2
        data_padded = np.pad(data, ((0, 0), (p, p), (p, p), (0, 0)))

        oh = int((data_height + p + p - k) / (self.stride) + 1)
        output_shape = n_data, oh, oh, self.n_filters

        self.data_padded = data_padded
        self.h_out = np.zeros(output_shape)

        for hi in range(oh):
            for wi in range(oh):
                h0 = hi * self.stride
                hf = h0 + k
                w0 = wi * self.stride
                wf = w0 + k
                self.h_out[:, hi, wi, :] = (data_padded[:, h0:hf, w0:wf, :, np.newaxis] * self.w[np.newaxis, :, :, :]).sum(axis=(1, 2, 3))

        return self.h_out / (self.h_out.max()/2)

    def has_w(self):
        return True

    def update(self, gradient_chain, learning_rate=.05, last=False):
        """

        :param gradient_chain: with shape (n_data, h_out, w_out, n_filters)
        """

        n_data, h_out, w_out, n_filters = gradient_chain.shape
        n_data, data_height, data_width, n_channels = self.h_in.shape
        k, k, _, n_filters = self.w.shape

        p = (k - 1) // 2
        padded_output = np.pad(np.zeros_like(self.h_in),
                               ((0,0), (p,p), (p,p), (0,0)))

        dw = np.zeros_like(self.w)

        for hi in range(h_out):
            for wi in range(w_out):
                h0 = hi * self.stride
                hf = h0 + k
                w0 = wi * self.stride
                wf = w0 + k

                padded_output[:, h0:hf, w0:wf, :] += (
                    np.sum(
                        self.w[np.newaxis, :, :, :, :] *
                        gradient_chain[:, hi:hi+1, wi:wi+1, np.newaxis, :],
                        axis=4
                        )
                )
                dw += (
                    np.sum(self.data_padded[:, h0:hf, w0:wf, :, np.newaxis] *
                           gradient_chain[:, hi:hi+1, wi:wi+1, np.newaxis, :],
                           axis=0)
                )
        dw = dw / (n_data)
        self.w = self.w - (learning_rate * dw)
        return padded_output[:, p:p + data_height, p:p+data_height, :]