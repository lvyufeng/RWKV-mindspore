import time
import unittest
import numpy as np
import mindspore
from mindspore import Tensor, ops
from src.layer.rwkv import RWKVLayer, BiRWKV
from src.layer.rwkv_custom_op import rwkv_forward, rwkv_backward

class TestRWKV(unittest.TestCase):
    # def test_rwkv_single_layer(self):
    #     net = RWKVLayer(20, 30)
    #     inputs = Tensor(np.random.randn(10, 32, 20), mindspore.float32)
    #     outputs, hidden = net(inputs)

    #     assert outputs.shape == (10, 32, 30)
    #     assert hidden.shape == (1, 32, 30)

    # def test_rwkv_bidirectional(self):
    #     net = BiRWKV(20, 30, bidirectional=True)
    #     inputs = Tensor(np.random.randn(10, 32, 20), mindspore.float32)
    #     outputs, hidden = net(inputs)

    #     assert outputs.shape == (10, 32, 60)
    #     assert hidden.shape == (2, 32, 30)

    def test_rwkv_custom_np_foward(self):
        k = np.random.randn(10, 32, 20)
        v = np.random.randn(10, 32, 20)
        w_u = np.random.randn(20,)
        w_w = np.random.randn(20,)

        start = time.time()
        output = rwkv_forward(k, v, w_w, w_u)
        end = time.time()
        print(end - start)
        # print(output)

    def test_rwkv_custom_op_foward(self):
        rwkv_op = ops.Custom(rwkv_forward,
                             out_shape=lambda k, v, w_w, w_u: k,
                             out_dtype=lambda k, v, w_w, w_u: k, func_type='akg')

        k = Tensor(np.random.randn(512, 32, 768), mindspore.float32)
        v = Tensor(np.random.randn(512, 32, 768), mindspore.float32)
        w_u = Tensor(np.random.randn(768,), mindspore.float32)
        w_w = Tensor(np.random.randn(768,), mindspore.float32)

        start = time.time()
        output = rwkv_op(k, v, w_w, w_u)
        end = time.time()
        print(end - start)
        # print(output)
        start = time.time()
        output = rwkv_op(k, v, w_w, w_u)
        end = time.time()
        print(end - start)



        assert output.shape == (512, 32, 768)


    def test_rwkv_custom_np_backward(self):
        k = np.random.randn(10, 32, 20)
        v = np.random.randn(10, 32, 20)
        w_u = np.random.randn(20,)
        w_w = np.random.randn(20,)
        gy = np.random.randn(10, 32, 20)

        start = time.time()
        gk, gv, gw, gu = rwkv_backward(k, v, w_w, w_u, gy)
        # print(gw, gu)
        # print(gk, gv)
        end = time.time()
        print(end - start)
        # print(output)

    def test_rwkv_custom_op_backward(self):
        def infer_shape(k, v, w_w, w_u, gy):
            return k, v, k[1:], k[1:]

        rwkv_backward_op = ops.Custom(rwkv_backward,
                             out_shape=infer_shape,
                             out_dtype=lambda k, v, w_w, w_u, gy: (k, v, k, k))

        k = Tensor(np.random.randn(512, 32, 768), mindspore.float32)
        v = Tensor(np.random.randn(512, 32, 768), mindspore.float32)
        w_u = Tensor(np.random.randn(768,), mindspore.float32)
        w_w = Tensor(np.random.randn(768,), mindspore.float32)
        gy = Tensor(np.random.randn(512, 32, 768), mindspore.float32)

        start = time.time()
        output = rwkv_backward_op(k, v, w_w, w_u, gy)
        end = time.time()
        print(end - start)
        print(output)
        # start = time.time()
        # output = rwkv_backward_op(k, v, w_w, w_u, gy)
        # end = time.time()
        # print(end - start)



    #     assert output.shape == (512, 32, 768)

    def test_rwkv_auto_diff(self):
        class RWKV(nn.Cell):
            def __init__(self):
                super().__init__()
                self.rwkv_op = ops.Custom(rwkv_forward,
                                    out_shape=lambda k, v, w_w, w_u: k,
                                    out_dtype=lambda k, v, w_w, w_u: k, func_type='hybrid')
            def construct(self, k, v, w_u, w_w):
                return self.rwkv_op(k, v, w_u, w_w)

        k = Tensor(np.random.randn(10, 32, 20), mindspore.float32)
        v = Tensor(np.random.randn(10, 32, 20), mindspore.float32)
        w_u = Tensor(np.random.randn(20,), mindspore.float32)
        w_w = Tensor(np.random.randn(20,), mindspore.float32)

        # @mindspore.jit
        # def forward(k, v, w_u, w_w):
        #     return rwkv_op(k, v, w_u, w_w)
        net = RWKV()
        
        # grad_fn = ops.value_and_grad(forward, (0, 1, 2, 3))
        grad_fn = ops.value_and_grad(net, (0, 1, 2, 3))
        print(grad_fn)
        out, grads = grad_fn(k, v, w_u, w_w)
        print(grads)