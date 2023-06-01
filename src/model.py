import math
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from .ops import load_wkv_cuda_kernel

RWKV_HEAD_QK_DIM = 0

class L2Wrap(nn.Cell):
    def construct(self, loss, y):
        return loss

    def bprop(self, loss, y, out, dout):
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = ops.max(y, -1, keepdims=True)
        gy = ops.zeros_like(y)
        gy = gy.scatter(-1, ids, maxx * factor)
        return (dout, gy)

class WKV(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.wkv_forward = load_wkv_cuda_kernel('wkv_forward', config.ctx_len)
        self.wkv_backward = load_wkv_cuda_kernel('wkv_backward', config.ctx_len)

    def construct(self, w, u, k, v):
        w = -ops.exp(w)
        y = self.wkv_forward(w, u, k, v)

        return y
    
    def bprop(self, w, u, k, v, y, gy):
        gw, gu, gk, gv = self.wkv_backward(w, u, k, v, gy)
        gw = ops.sum(gw, 0)
        gu = ops.sum(gu, 0)

        return (gw, gu, gk, gv)

class RWKV_TimeMix(nn.Cell):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        ratio_0_to_1 = (layer_id / (config.n_layer - 1)) # 0 to 1
        ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0
        
        # fancy time_decay
        decay_speed = np.ones(attn_sz)
        for h in range(attn_sz):
            decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = Parameter(Tensor(decay_speed, mindspore.float32), 'time_decay')
        # fancy time_first
        zigzag = (np.array([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
        self.time_first = Parameter(Tensor(np.ones(attn_sz) * math.log(0.3) + zigzag, mindspore.float32), 'time_first')
            
            # fancy time_mix
        x = np.ones((1, 1, config.n_embd))
        for i in range(config.n_embd):
            x[0, 0, i] = i / config.n_embd
        self.time_mix_k = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float32), 'time_mix_k')
        self.time_mix_v = Parameter(Tensor(np.power(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1, mindspore.float32), 'time_mix_v')
        self.time_mix_r = Parameter(Tensor(np.power(x, 0.5 * ratio_1_to_almost0), mindspore.float32), 'time_mix_r')
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Dense(config.n_embd, attn_sz, has_bias=False)
        self.value = nn.Dense(config.n_embd, attn_sz, has_bias=False)
        self.receptance = nn.Dense(config.n_embd, attn_sz, has_bias=False)

        self.output = nn.Dense(attn_sz, config.n_embd, has_bias=False)

        self.wkv = WKV(config)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def construct(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = ops.sigmoid(r)

        rwkv = sr * self.wkv(self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv

class RWKV_ChannelMix(nn.Cell):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0

        x = np.ones((1, 1, config.n_embd))
        for i in range(config.n_embd):
            x[0, 0, i] = i / config.n_embd

        self.time_mix_k = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float32), 'time_mix_k')
        self.time_mix_r = Parameter(Tensor(np.power(x, ratio_1_to_almost0), mindspore.float32), 'time_mix_r')

        hidden_sz = 4 * config.n_embd
        self.key = nn.Dense(config.n_embd, hidden_sz, has_bias=False)
        self.receptance = nn.Dense(config.n_embd, config.n_embd, has_bias=False)
        self.value = nn.Dense(hidden_sz, config.n_embd, has_bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def construct(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = ops.square(ops.relu(k))
        kv = self.value(k)

        rkv = ops.sigmoid(self.receptance(xr)) * kv
        return rkv


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Cell):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm([config.n_embd], epsilon=1e-5)
        self.ln2 = nn.LayerNorm([config.n_embd], epsilon=1e-5)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm([config.n_embd], epsilon=1e-5)

        if self.layer_id == 0 and config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, 0)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)
        self.model_type = config.model_type

    def construct(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and self.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # better in some cases
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.SequentialCell(*[Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm([config.n_embd], epsilon=1e-5)
        self.head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)

        self.l2_wrapper = L2Wrap()

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Dense(config.n_embd, RWKV_HEAD_QK_DIM, has_bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Dense(config.n_embd, RWKV_HEAD_QK_DIM, has_bias=False)
            self.head_k.scale_init = 0.1
            self.copy_mask = Tensor(np.tril(np.ones(config.ctx_len, config.ctx_len)), mindspore.float32)

        self.ctx_len = config.ctx_len
        self.vocab_size = config.vocab_size

    def get_ctx_len(self):
        return self.ctx_len

    def construct(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = ops.matmul(q, k.swapaxes(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)            
            c = ops.matmul(c, ops.one_hot(idx, self.vocab_size, 1, 0))
            x = self.head(x) + c
        else:
            x = self.head(x)

        loss = None
        if targets is not None:
            loss = ops.cross_entropy(x.view(-1, x.shape[-1]), targets.view(-1))
        if loss is not None:
            return self.l2_wrapper(loss, x)
        return x
