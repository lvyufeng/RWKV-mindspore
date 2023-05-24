import os
import mindspore
from mindspore import ops

device_target = mindspore.get_context('device_target')
if device_target == 'GPU':
    so_path = os.path.dirname(os.path.realpath(__file__)) + '/cuda/wkv.so'
    # str(os.path. .('./cuda/wkv.so'))
elif device_target == 'Ascend':
    so_path = os.path.dirname(os.path.realpath(__file__)) + '/bisheng/wkv.so'

wkv_forward_op = ops.Custom(
    so_path + ':wkv_forward',
    out_shape=lambda w, u, k, v: k,
    out_dtype=lambda w, u, k, v: k,
    func_type='aot'
)

def infer_shape(w_w, w_u, k, v, gy):
    return (k[0], k[2]), (k[0], k[2]), k, v

wkv_backward_op = ops.Custom(
    so_path + ':wkv_backward',
    out_shape=infer_shape,
    out_dtype=lambda w_w, w_u, k, v, gy: (k, k, k, k),
    # out_shape=(),
    # out_dtype=(),
    func_type='aot'
)

wkv_forward_op.add_prim_attr('primitive_target', device_target)
wkv_backward_op.add_prim_attr('primitive_target', device_target)
