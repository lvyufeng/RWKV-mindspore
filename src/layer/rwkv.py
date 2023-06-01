import math
import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Orthogonal, Normal, Uniform
from mindspore.ops._primitive_cache import _get_cache_prim
from ..ops import load_wkv_cuda_kernel

class WKV(nn.Cell):
    def __init__(self, ctx_len):
        super().__init__()
        # self.wkv_forward = load_wkv_cuda_kernel('wkv_forward_with_state', ctx_len)
        self.wkv_forward = load_wkv_cuda_kernel('wkv_forward', ctx_len)
        self.wkv_backward = load_wkv_cuda_kernel('wkv_backward', ctx_len)

    # def construct(self, w, u, k, v, s):
    #     # w = -ops.exp(w)
    #     y = self.wkv_forward(w, u, k, v, s)

    #     return y
    def construct(self, w, u, k, v):
        w = -ops.exp(w)
        y = self.wkv_forward(w, u, k, v)

        return y
    
    def bprop(self, w, u, k, v, y, gy):
        gw, gu, gk, gv = self.wkv_backward(w, u, k, v, gy)
        gw = ops.sum(gw, 0)
        gu = ops.sum(gu, 0)
        return (gw, gu, gk, gv)

class RWKVLayer(nn.Cell):
    def __init__(self, seq_len, input_size, hidden_size):
        super().__init__(False)
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = 1 / math.sqrt(hidden_size)
        self.w_r = Parameter(initializer(Uniform(std), (hidden_size, input_size)), 'w_r')
        self.w_k = Parameter(initializer(Uniform(std), (hidden_size, input_size)), 'w_k')
        self.w_v = Parameter(initializer(Uniform(std), (hidden_size, input_size)), 'w_v')
        # self.w_o = Parameter(initializer(Normal(), (hidden_size, input_size)), 'w_o')
        self.w_w = Parameter(initializer(Uniform(std), (hidden_size)), 'w_w')
        self.w_u = Parameter(initializer(Uniform(std), (hidden_size)), 'w_u')
        self.wkv = WKV(seq_len)

    def construct(self, inputs, frac_n=None, frac_d=None, scale=None):
        batch_size, seq_length, _ = inputs.shape

        # if frac_n is None:
        #     frac_n = Tensor(np.zeros((batch_size, self.hidden_size)), mindspore.float32)
        # if frac_d is None:
        #     frac_d = Tensor(np.zeros((batch_size, self.hidden_size)), mindspore.float32)
        # if scale is None:
        #     scale = Tensor(np.full((batch_size, self.hidden_size), -1e20), mindspore.float32)

        r = ops.sigmoid(ops.matmul(inputs, self.w_r.swapaxes(0, 1))) # (batch_size, seq_length, hidden_size)
        k = ops.matmul(inputs, self.w_k.swapaxes(0, 1)) # (batch_size, seq_length, hidden_size)
        v = ops.matmul(inputs, self.w_v.swapaxes(0, 1)) # (batch_size, seq_length, hidden_size)

        # output = self.wkv(self.w_w, self.w_u, k, v, ops.stack([frac_n, frac_d, scale], 2))
        output = self.wkv(self.w_w, self.w_u, k, v)

        # output = ops.matmul(output, self.w_o.swapaxes(0, 1))
        # output = ops.matmul(r * output, self.w_o)
        output = r * output
        return output, output[:, -1, :].expand_dims(0)


class BiRWKV(nn.Cell):
    def __init__(self, seq_length, input_size, hidden_size, batch_first=False, bidirectional=False, dropout=0.0):
        super().__init__(False)
        self.fw = RWKVLayer(seq_length, input_size, hidden_size)
        self.bidirecitonal = bidirectional
        self.batch_first = batch_first
        if bidirectional:
            self.bw = RWKVLayer(seq_length, input_size, hidden_size)
        else:
            self.bw = None
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, inputs, frac_n=None, frac_d=None, scale=None):
        if not self.batch_first:
            inputs = inputs.swapaxes(0, 1)
        output_fw, hidden_fw = self.fw(inputs, frac_n, frac_d, scale)
        if self.bidirecitonal:
            inputs_fw = ops.reverse(inputs, (1,))
            output_bw, hidden_bw = self.bw(inputs_fw, frac_n, frac_d, scale)
            output_bw = ops.reverse(output_bw, (1,))
            output = ops.concat((output_fw, output_bw), 2)
            hidden = ops.concat((hidden_fw, hidden_bw), 0)
        else:
            output = output_fw
            hidden = hidden_fw.unsqueeze(0)
        if not self.batch_first:
            output = output.swapaxes(0, 1)
        return output, hidden
