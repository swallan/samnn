import numpy as np

class MaxPool():
    def apply(self, x):
        pool_param = {'pool_height': 4, 'pool_width': 4, 'stride':1}
        """
        A naive implementation of the forward pass for a max pooling layer.
        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass                              #
        #############################################################################
        (N, C, H, W) = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        H_prime = int(1 + (H - pool_height) / stride)
        W_prime = int(1 + (W - pool_width) / stride
)
        out = np.zeros((N, C, H_prime, W_prime))

        for n in range(N):
            for h in range(H_prime):
                for w in range(W_prime):
                    h1 = h * stride
                    h2 = h * stride + pool_height
                    w1 = w * stride
                    w2 = w * stride + pool_width
                    window = x[n, :, h1:h2, w1:w2]
                    out[n, :, h, w] = np.max(
                        window.reshape((C, pool_height * pool_width)), axis=1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = (x, pool_param)
        self.cache = cache
        return out

    def update(self, dout):
        """
        A naive implementation of the backward pass for a max pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """

        dx = None
        #############################################################################
        # TODO: Implement the max pooling backward pass                             #
        #############################################################################
        (x, pool_param) = self.cache
        (N, C, H, W) = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        H_prime = 1 + (H - pool_height) / stride
        W_prime = 1 + (W - pool_width) / stride

        dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h in range(H_prime):
                    for w in range(W_prime):
                        h1 = h * stride
                        h2 = h * stride + pool_height
                        w1 = w * stride
                        w2 = w * stride + pool_width
                        window = x[n, c, h1:h2, w1:w2]
                        window2 = np.reshape(window,
                                             (pool_height * pool_width))
                        window3 = np.zeros_like(window2)
                        window3[np.argmax(window2)] = 1

                        dx[n, c, h1:h2, w1:w2] = np.reshape(window3, (
                        pool_height, pool_width)) * dout[n, c, h, w]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx
