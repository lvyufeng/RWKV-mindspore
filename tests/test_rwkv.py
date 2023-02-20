import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from src.layer.rwkv import RWKVLayer, BiRWKV

class TestRWKV(unittest.TestCase):
    def test_rwkv_single_layer(self):
        net = RWKVLayer(20, 30)
        inputs = Tensor(np.random.randn(10, 32, 20), mindspore.float32)
        outputs, hidden = net(inputs)

        assert outputs.shape == (10, 32, 30)
        assert hidden.shape == (1, 32, 30)

    def test_rwkv_bidirectional(self):
        net = BiRWKV(20, 30, bidirectional=True)
        inputs = Tensor(np.random.randn(10, 32, 20), mindspore.float32)
        outputs, hidden = net(inputs)

        assert outputs.shape == (10, 32, 60)
        assert hidden.shape == (2, 32, 30)
