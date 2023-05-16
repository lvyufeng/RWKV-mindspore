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

@pytest.mark.parametrize('jit', [True, False])
def test_backward(jit):
    config = GPTConfig(1000, 512, model_type='RWKV', n_layer=2, n_embd=128)
    model = GPT(config)
    
    input_ids = Tensor(np.random.randint(0, 1000, (2, 32)), mindspore.int32)
    def forward(input_ids, labels):
        loss = model(input_ids, labels)
        return loss

    grad_fn = mindspore.value_and_grad(forward, None, model.trainable_params())

    def train_step(input_ids, labels):
        loss, grads = grad_fn(input_ids, labels)
        return loss
    if jit:
        train_step = ms_jit(train_step)

    loss = train_step(input_ids[:-1], input_ids[1:])
    print(loss)