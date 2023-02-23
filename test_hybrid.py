
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import kernel

@kernel
def outer_product(a, b):
    c = output_tensor(a.shape, a.dtype)

    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         for k in range(a.shape[2]):
    #             c[i, j, k] = a[i, j, k] + b[i, j, k]
    return c

np_x = np.random.normal(0, 1, [4, 4, 4]).astype(np.float32)
np_y = np.random.normal(0, 1, [4, 4, 4]).astype(np.float32)

print(outer_product(np_x, np_y))

input_x = ms.Tensor(np_x)
input_y = ms.Tensor(np_y)

test_op_akg = ops.Custom(outer_product)
out = test_op_akg(input_x, input_y)
print(out)
