import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Orthogonal
from mindspore.ops._primitive_cache import _get_cache_prim

def matmul(a, b, transpose_a=False, transpose_b=False):
    _matmul = _get_cache_prim(ops.MatMul)(transpose_a, transpose_b)
    a_shape = a.shape[:-1]
    a = a.view(-1, a.shape[-1])
    out = _matmul(a, b)
    return out.view(a_shape + (-1,))

class RWKVCellV3(nn.Cell):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_r = Parameter(initializer(Orthogonal(), (input_size, hidden_size)), 'w_r')
        self.w_k = Parameter(initializer(Orthogonal(), (input_size, hidden_size)), 'w_k')
        self.w_v = Parameter(initializer(Orthogonal(), (input_size, hidden_size)), 'w_v')
        self.w_o = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_o')
        self.w_u = Parameter(initializer('ones', (hidden_size)), 'w_u')
        self.w_w = Parameter(initializer('ones', (hidden_size)), 'w_w')

    def construct(self, inputs, a, b):
        r = ops.sigmoid(ops.matmul(inputs, self.w_r))
        k = ops.exp(ops.matmul(inputs, self.w_k))
        v = ops.matmul(inputs, self.w_v)
        c_t = a + self.w_u * k * v
        d_t = b + self.w_u * k
        a_n = self.w_w * a + k * v
        b_n = self.w_w * b + k

        o = ops.matmul(r * (c_t / d_t), self.w_o) 

        return o, a_n, b_n


class RWKVCellV4(nn.Cell):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_r = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_r')
        self.w_k = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_k')
        self.w_v = Parameter(initializer(Orthogonal(), (hidden_size, input_size)), 'w_v')
        self.w_o = Parameter(initializer(Orthogonal(), (input_size, input_size)), 'w_o')
        self.w_u = Parameter(initializer('ones', (hidden_size,)), 'w_u')
        self.w_w = Parameter(initializer('ones', (hidden_size,)), 'w_w')

    def construct(self, inputs, frac_n, frac_d, scale):
        """
        args:
            inputs: (batch_size, input size).
            frac_n: (batch_size, hidden_size), numerator of softmax, must be a zeros at first step.
            frac_d: (batch_size, hidden_size), denominator of softmax, must be a zeros at first step.
            scale: (batch_size, hidden_size), scale value passing through time for overflow currection,
                must be a minimum value at first step, like -1e20.
        """
        # t indices time step.
        r_t = ops.sigmoid(matmul(inputs, self.w_r, transpose_b=True)) # (batch_size, hidden_size)
        k_t = matmul(inputs, self.w_k, transpose_b=True) # (batch_size, hidden_size)
        v_t = matmul(inputs, self.w_v, transpose_b=True) # (batch_size, hidden_size)

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

        output = matmul(r_t * (frac_n_t / frac_d_t), self.w_o, transpose_b=True)

        return output, next_frac_n, next_frac_d, next_scale
