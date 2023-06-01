import mindspore
from mindspore import ops
from mindspore import log as logger

WKV_SHAPE_INFER = {
    'wkv_forward': lambda w, u, k, v: k,
    'wkv_forward_with_state': lambda w, u, k, v, s: k,
    'wkv_backward': lambda w_w, w_u, k, v, gy: ((k[0], k[2]), (k[0], k[2]), k, v),
}

WKV_DTYPE_INFER = {
    'wkv_forward': lambda w, u, k, v: k,
    'wkv_forward_with_state': lambda w, u, k, v, s: k,
    'wkv_backward': lambda w_w, w_u, k, v, gy: (w_w, w_u, k, v),
}

def load_wkv_cuda_kernel(func_name, context_length):
    """load wkv cuda kernel"""
    device_target = mindspore.get_context('device_target')
    if device_target != 'GPU':
        raise RuntimeError('WKV operator only support GPU currently.')

    logger.info(f"Loading CUDA kernel for RWKV at context length of {context_length}.")

    from .utils import compile_kernel
    so_path = compile_kernel(Tmax=context_length)
    wkv_op = ops.Custom(
        str(so_path) + ':' + func_name,
        out_shape=WKV_SHAPE_INFER[func_name],
        out_dtype=WKV_DTYPE_INFER[func_name],
        func_type='aot'
    )
    wkv_op.add_prim_attr('primitive_target', device_target)
    return wkv_op

