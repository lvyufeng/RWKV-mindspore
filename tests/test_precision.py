import os
import numpy as np
import torch
import mindspore
from mindspore import Tensor

from src.model import GPT, GPTConfig, WKV
from pt_src.src.model import GPT as ptGPT
from pt_src.src.model import WKV as ptWKV


os.environ['RWKV_FLOAT_MODE'] = 'fp32'

def test_wkv_precision():
    batch_size = 2
    seq_length = 1024
    hidden_size = 512
    w = np.random.randn(hidden_size).astype(np.float32)
    u = np.random.randn(hidden_size).astype(np.float32)
    k = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    v = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)

    msWKV = WKV(GPTConfig(100, 1024))
    ms_y = msWKV(Tensor(w), Tensor(u), Tensor(k), Tensor(v))
    pt_y = ptWKV.apply(batch_size, seq_length, hidden_size,
                       torch.tensor(w).cuda(),
                       torch.tensor(u).cuda(),
                       torch.tensor(k).cuda(),
                       torch.tensor(v).cuda())

    assert np.allclose(ms_y.asnumpy(), pt_y.cpu().detach().numpy())

def test_random_init_precision():
    n_layer = 12
    n_embed = 1024
    ctx_len = 1024

    config = GPTConfig(50277, ctx_len, model_type='RWKV', n_layer=n_layer, n_embd=n_embed)
    ms_model = GPT(config)
    pt_model = ptGPT(config)

    state_dict = pt_model.state_dict()
    ms_dict = ms_model.parameters_and_names()

    for k, v in ms_dict:
        if 'embedding_table' in k:
            k = k.replace('embedding_table', 'weight')
        if 'gamma' in k:
            k = k.replace('gamma', 'weight')
        if 'beta' in k:
            k = k.replace('beta', 'bias')

        v.set_data(Tensor(state_dict[k].detach().cpu().numpy()))

    input_ids = np.random.randint(0, 50277, (2, 1024))

    ms_model.set_train(False)
    pt_model.eval()

    ms_out = ms_model(Tensor(input_ids))
    ms_out = ms_out.asnumpy()

    pt_model.to('cuda')
    pt_out = pt_model(torch.tensor(input_ids).to('cuda'))


    print(pt_out)

    assert np.allclose(ms_out, pt_out.detach().cpu().numpy(), 1e-3, 1e-3)