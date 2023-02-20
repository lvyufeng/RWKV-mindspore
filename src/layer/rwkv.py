import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Orthogonal
from mindspore.ops._primitive_cache import _get_cache_prim
from .rwkv_cell import matmul

class RWKVLayer(nn.Cell):
    def __init__(self, input_size, hidden_size):
        super().__init__(False)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_r = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_r')
        self.w_k = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_k')
        self.w_v = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_v')
        # self.w_o = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_o')
        self.w_u = Parameter(initializer('ones', (hidden_size)), 'w_u')
        self.w_w = Parameter(initializer('ones', (hidden_size)), 'w_w')

    def construct(self, inputs, frac_n=None, frac_d=None, scale=None):
        seq_length, batch_size, _ = inputs.shape

        if frac_n is None:
            frac_n = Tensor(np.zeros((batch_size, self.hidden_size)), mindspore.float32)
        if frac_d is None:
            frac_d = Tensor(np.zeros((batch_size, self.hidden_size)), mindspore.float32)
        if scale is None:
            scale = Tensor(np.full((batch_size, self.hidden_size), -1e20), mindspore.float32)

        r = ops.sigmoid(matmul(inputs, self.w_r, transpose_b=True)) # (seq_length, batch_size, hidden_size)
        k = matmul(inputs, self.w_k, transpose_b=True) # (seq_length, batch_size, hidden_size)
        v = matmul(inputs, self.w_v, transpose_b=True) # (seq_length, batch_size, hidden_size)

        t = Tensor(0, mindspore.int32)
        output = Tensor(np.zeros((seq_length, batch_size, self.hidden_size)), mindspore.float32)
        while t < seq_length:
            k_t = k[t]
            v_t = v[t]
            r_t = r[t]
            softmax_scale = ops.maximum(scale, self.w_u + k_t) # input scale for softmax to fix overflow
            cell_multiplier = ops.exp(scale - softmax_scale)
            attn_multiplier = ops.exp(self.w_u + k_t - softmax_scale)
            frac_n_t = cell_multiplier * frac_n + attn_multiplier * v_t
            frac_d_t = cell_multiplier * frac_d + attn_multiplier

            next_scale = ops.maximum(scale + self.w_w, k_t) # input scale for passing cell to fix overflow
            next_cell_multiplier = ops.exp(scale + self.w_w - next_scale)
            next_attn_multiplier = ops.exp(k_t - next_scale)
            next_frac_n = next_cell_multiplier * frac_n + next_attn_multiplier * v_t
            next_frac_d = next_cell_multiplier * frac_d + next_attn_multiplier
            y = r_t * (frac_n_t / frac_d_t)
            output[t] = y
            t += 1
            scale = next_scale
            frac_n = next_frac_n
            frac_d = next_frac_d

        # output = matmul(output, self.w_o, transpose_b=True)
        return output, y.unsqueeze(0)

class BiRWKV(nn.Cell):
    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super().__init__(False)
        self.fw = RWKVLayer(input_size, hidden_size)
        self.bidirecitonal = bidirectional
        self.batch_first = batch_first
        if bidirectional:
            self.bw = RWKVLayer(input_size, hidden_size)
        else:
            self.bw = None
    
    def construct(self, inputs, frac_n=None, frac_d=None, scale=None):
        if self.batch_first:
            inputs = inputs.swapaxes(0, 1)
        output_fw, hidden_fw = self.fw(inputs, frac_n, frac_d, scale)
        if self.bidirecitonal:
            inputs_fw = ops.reverse(inputs, (0,))
            output_bw, hidden_bw = self.bw(inputs_fw, frac_n, frac_d, scale)
            output_bw = ops.reverse(output_bw, (0,))
            output = ops.concat((output_fw, output_bw), 2)
            hidden = ops.concat((hidden_fw, hidden_bw), 0)
        else:
            output = output_fw
            hidden = hidden_fw.unsqueeze(0)
        if self.batch_first:
            output = output.swapaxes(0, 1)
        return output, hidden