
import numpy as np

def conv_im2col(input, w):
    """Optimized Convolutions via matrix multiplication

    For an input image of shape (10, 10, 3) and a N kernels to be
    convolved with shape (5, 5, 3). The kernels each have the same
    depth (number of channels) as the input data.

    So we need to take blocks of pixels of shape (5, 5, 3) out of the
    image and stretch each of them into a column (length 5x5x3 = 75).
    For stride=1 and padding=0, this means that there will be 100 rows
    of 75 columns.

    ```data_reshaped.shape = (100, 75)```

    If there are N=7 kernels of shape (5, 5, 3), then we need to reshape
    it to have 7 columns of flattened kernels, 75.

    ```w_reshaped.shape = (75 x 7).

    Dot them together to get resulting shape:

    (100, 75) @ (75 x 7) = (100 x 7).

    This now can be reshaped back to the image size and the new number of channels, 7.

    Result shape: (10 x 10 x 7).

    :param input: feature map of H x W x C
    :param w: conv filter of N x K x K x C

    Source: https://youtu.be/pA4BsUK3oP4?t=2241,
            https://cs231n.github.io/convolutional-networks/
    """

    # height, width, channels
    ND, C, H, W = input.shape
    # number of kernels, kernel height, kernel width, channels
    NK, C, K, K = w.shape

    stride = 1
    padding = None

    # with no padding and stride=1, the number of blocks in each dimension
    block_height_n = int((H - K) / stride + 1)
    block_width_n = int((W - K) / stride + 1)

    # we need to put zero padding around the array so that it does not go out of bounds.
    hbw = int(np.floor(K / 2))
    input_padded = np.pad(input, ((0,0), (0, 0), (hbw, hbw), (hbw, hbw)))

    # grab each block from the input data.
    input_blocks_as_rows = []
    for row in range(0, input_padded.shape[2] - hbw - 2, stride):
        for col in range(0, input_padded.shape[3] - hbw - 2, stride):
            block = input_padded[:, :, row:row + K, col: col + K]
            input_blocks_as_rows.append(block.flatten())
    input_blocks_as_col = np.transpose(input_blocks_as_rows)

    # Since the input data is now in column form, we need to put the data
    # for the weights into rows.
    w_reshaped = w.reshape(NK, K * K * C)

    # the dot product of these two can be reshaped to the proper output
    # dimension
    res = w_reshaped @ input_blocks_as_col.reshape(ND, K**2 * n_channel, W*H)
    print('w_reshaped',w_reshaped.shape, 'input_blocks_as_col.reshape(ND, K**2 * n_channel, W*H)',input_blocks_as_col.reshape(ND, K**2 * n_channel, W*H).shape)
    print('res', res.shape)
    out_height = int((H + 2 * 2 - K) / stride + 1)
    out_width = int((W + 2 * 2 - K) / stride + 1)
    print(res.shape)
    res_reshaped = res.reshape(ND, n_filter, out_height, out_width)
    return res_reshaped, (C, H, W, res.shape, input_blocks_as_col, input)

def conv_im2col_backprop(grad_chain, w, cache, lr=.01):
    # the gradient calculation for this is the same as a fully connected layer
    # once you have the matrices in the correct shapes.
    C, H, W, res_shape, input_blocks_as_col, input = cache
    NK, C, K, K = w.shape

    # reshape to be n_filters x h x w
    grad_reshaped = grad_chain.reshape(grad_chain.shape[0], # n_data
                                       NK, # n_channel
                                       grad_chain.shape[2] ** 2)
    input_blocks_as_col_reshaped = input_blocks_as_col.reshape(k * k, H*W, input.shape[0])
    grad_wrt_w = grad_reshaped @ input_blocks_as_col_reshaped.T
    grad_wrt_w_reshaped = grad_wrt_w.reshape(w.shape)
    w = w - (grad_wrt_w_reshaped * lr)

    w_reshaped = w.reshape(NK, K * K * C)

    grad_chain = w_reshaped.T @ grad_reshaped

    grad_matrix_out = np.zeros((C, H, W))
    hbw = int(np.floor(K / 2))
    output_padded = np.pad(grad_matrix_out, ((0, 0), (hbw, hbw), (hbw, hbw)))
    blocks_as_rows = input_blocks_as_col.T

    for i in range(len(blocks_as_rows)):
        block = blocks_as_rows[i].reshape(3, 5, 5)
        # need to figure out where it goes in the picture.
        # row x col of upper left corder
        row = int(np.floor(i / H))
        col = int(i % W)
        output_padded[:, row:row + 5, col:col + 5] = block
    output_unpadded = output_padded[:, 2:30, 2:30]

    return output_unpadded

n_data, n_channel, h, w = 10, 1, 28, 28
n_filter, k, = 1, 5
data = np.arange(n_data * n_channel * h * w).reshape(n_data, n_channel, h, w)
filter = np.zeros((n_filter, n_channel, k, k))
filter[:,:,2:3,2:3] = 1
o, cache = conv_im2col(data, filter)
print(o.shape)
assert np.all(o == data)
print(conv_im2col_backprop(o, filter, cache))












# def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
#     # First figure out what the size of the output should be
#     N, C, H, W = x_shape
#     assert (H + 2 * padding - field_height) % stride == 0
#     assert (W + 2 * padding - field_height) % stride == 0
#     out_height = int((H + 2 * padding - field_height) / stride + 1)
#     out_width = int((W + 2 * padding - field_width) / stride + 1)
#
#     i0 = np.repeat(np.arange(field_height), field_width)
#     i0 = np.tile(i0, C)
#     i1 = stride * np.repeat(np.arange(out_height), out_width)
#     j0 = np.tile(np.arange(field_width), field_height * C)
#     j1 = stride * np.tile(np.arange(out_width), out_height)
#     i = i0.reshape(-1, 1) + i1.reshape(1, -1)
#     j = j0.reshape(-1, 1) + j1.reshape(1, -1)
#
#     k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
#
#     return (k.astype(int), i.astype(int), j.astype(int))
#
#
# def im2col_indices(x, field_height, field_width, padding=1, stride=1):
#     """ An implementation of im2col based on some fancy indexing """
#     # Zero-pad the input
#     p = padding
#     x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
#
#     k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
#
#     cols = x_padded[:, k, i, j]
#     C = x.shape[1]
#     cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
#     return cols
#
#
# def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
#                    stride=1):
#     """ An implementation of col2im based on fancy indexing and np.add.at """
#     N, C, H, W = x_shape
#     H_padded, W_padded = H + 2 * padding, W + 2 * padding
#     x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
#     k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
#     cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
#     cols_reshaped = cols_reshaped.transpose(2, 0, 1)
#     np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
#     if padding == 0:
#         return x_padded
#     return x_padded[:, :, padding:-padding, padding:-padding]
