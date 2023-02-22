import unittest
import time
import numpy as np
import mindspore
from mindspore import Tensor, ops
from src.layer.rwkv_custom_op import rwkv_cell

class TestRWKV(unittest.TestCase):
    def test_rwkv_custom_np(self):
        k = np.random.randn(10, 20)
        v = np.random.randn(10, 20)
        f_n = np.random.randn(10, 20)
        f_d = np.random.randn(10, 20)
        s = np.random.randn(10, 20)
        w_u = np.random.randn(20,)
        w_w = np.random.randn(20,)

        start = time.time()
        output, f_n_n, f_d_n, s_n = rwkv_cell(k, v, f_n, f_d, s, w_u, w_w)
        end = time.time()
        print(end - start)

        assert output.shape == (10, 20)

    def test_rwkv_custom_op(self):
        rwkv = ops.Custom(rwkv_cell,
                          out_shape=lambda k, v, f_n, f_d, s, w_u, w_w: (k, f_n, f_d, s),
                          out_dtype=lambda k, v, f_n, f_d, s, w_u, w_w: (k, f_n, f_d, s))

        k = Tensor(np.random.randn(10, 20), mindspore.float32)
        v = Tensor(np.random.randn(10, 20), mindspore.float32)
        f_n = Tensor(np.random.randn(10, 20), mindspore.float32)
        f_d = Tensor(np.random.randn(10, 20), mindspore.float32)
        s = Tensor(np.random.randn(10, 20), mindspore.float32)
        w_u = Tensor(np.random.randn(20,), mindspore.float32)
        w_w = Tensor(np.random.randn(20,), mindspore.float32)

        start = time.time()
        output, f_n_n, f_d_n, s_n = rwkv(k, v, f_n, f_d, s, w_u, w_w)
        end = time.time()
        print(end - start)
        start = time.time()
        output, f_n_n, f_d_n, s_n = rwkv(k, v, f_n, f_d, s, w_u, w_w)
        end = time.time()
        print(end - start)


        assert output.shape == (10, 20)
        # print(output.shape)
        # assert hidden.shape == (1, 32, 30)