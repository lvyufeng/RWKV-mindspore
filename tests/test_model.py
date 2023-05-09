import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import jit as ms_jit
from src.model import GPT, GPTConfig


@pytest.mark.parametrize('jit', [True, False])
def test_forward(jit):
    config = GPTConfig(1000, 512, model_type='RWKV', n_layer=2, n_embd=128)
    model = GPT(config)
    
    input_ids = Tensor(np.random.randint(0, 1000, (2, 32)), mindspore.int32)
    def forward(input_ids):
        outputs = model(input_ids)
        return outputs

    if jit:
        forward = ms_jit(forward)

    outputs = forward(input_ids)
    print(outputs.shape)
