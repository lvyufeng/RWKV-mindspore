
import numpy as np
import mindspore as ms
from mindspore.ops import ms_kernel

@ms_kernel
def rwkv_cell(k, v, frac_n, frac_d, scale, w_u, w_w):
    output = output_tensor(k.shape, k.dtype)
    next_frac_n = output_tensor(frac_n.shape, frac_n.dtype)
    next_frac_d = output_tensor(frac_d.shape, frac_d.dtype)
    next_scale = output_tensor(scale.shape, scale.dtype)
    softmax_scale = allocate(scale.shape, scale.dtype)

    for b in parallel(k.shape[0]):
        for c in vectorize(k.shape[1]):
            softmax_scale[b, c] = scale[b, c] if scale[b, c] > w_u[c] + k[b, c] else w_u[c] + k[b, c]
            output[b, c] = (exp(scale[b, c] - softmax_scale[b, c]) * frac_n[b, c] + exp(w_u[c] + k[b, c] - softmax_scale[b, c]) * v[b, c]) / \
                (exp(scale[b, c] - softmax_scale[b, c]) * frac_d[b, c] + exp(w_u[c] + k[b, c] - softmax_scale[b, c]))
            next_scale[b, c] = scale[b, c] + w_w[c] if scale[b, c] + w_w[c] > k[b, c] else k[b, c]
            next_frac_n[b, c] = exp(scale[b, c] + w_w[c] - next_scale[b, c]) * frac_n[b, c] + exp(k[b, c] - next_scale[b, c]) * v[b, c]
            next_frac_d[b, c] = exp(scale[b, c] + w_w[c] - next_scale[b, c]) * frac_d[b, c] + exp(k[b, c] - next_scale[b, c])

    return output, next_frac_n, next_frac_d, next_scale

