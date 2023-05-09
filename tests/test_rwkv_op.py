import time
import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from src.ops.wkv_op import wkv_forward_op, wkv_backward_op

class TestRWKV(unittest.TestCase):
    def test_rwkv_custom_op_forward(self):
        k = Tensor(np.random.randn(32, 512, 768), mindspore.float32)
        v = Tensor(np.random.randn(32, 512, 768), mindspore.float32)
        w_u = Tensor(np.random.randn(768,), mindspore.float32)
        w_w = Tensor(np.random.randn(768,), mindspore.float32)

        start = time.time()
        output = wkv_forward_op(w_w, w_u, k, v)
        end = time.time()
        print(end - start)
        # print(output)
        start = time.time()
        output = wkv_forward_op(w_w, w_u, k, v)
        end = time.time()
        print(end - start)
        # print(output)
        assert output.shape == (32, 512, 768)

    def test_rwkv_custom_op_backward(self):
        k = Tensor(np.random.randn(32, 512, 768), mindspore.float32)
        v = Tensor(np.random.randn(32, 512, 768), mindspore.float32)
        w_u = Tensor(np.random.randn(768,), mindspore.float32)
        w_w = Tensor(np.random.randn(768,), mindspore.float32)
        gy = Tensor(np.random.randn(32, 512, 768), mindspore.float32)

        start = time.time()
        gw, gu, gk, gv = wkv_backward_op(w_w, w_u, k, v, gy)
        end = time.time()
        print(end - start)
        start = time.time()
        gw, gu, gk, gv = wkv_backward_op(w_w, w_u, k, v, gy)
        end = time.time()
        print(gw, gu, gk, gv)
        print(gw.shape, gu.shape, gk.shape, gv.shape)
        print(end - start)
