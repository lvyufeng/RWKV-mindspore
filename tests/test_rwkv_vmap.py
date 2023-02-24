import mindspore as ms
from mindspore import ops, Tensor
ms.set_context(device_target='CPU', max_call_depth=100000)

def rwkv_time(k, v, w, u, p, q, o):
    """
    k: (t,)
    v: (t,)
    w: ()
    u: ()
    p: ()
    q: ()
    o: ()
    """
    # output = ops.zeros(k.shape)
    output = ()
    seq_len = k.shape[0]
    t = Tensor(0, ms.int32)
    # while t < seq_len:
    for t in range(k.shape[0]):
        # print(u.dtype)
        no = ops.maximum(o, k[t] + u)
        a = ops.exp(o - no)
        b = ops.exp(u + k[t] - no)
        # output[t] = (a * p + b * v[t]) / (a * q) + b
        y = (a * p + b * v[t]) / (a * q) + b
        output += (y,)

        no = ops.maximum(w + o, k[t])
        a = ops.exp(w + o - no)
        b = ops.exp(k[t] - no)
        p = a * p + b * v[t]
        q = a * q + b
        o = no
        # t += 1
    # print(output)
    return ops.stack(output, 0)
    # return output

b = 32
t = 100
c = 768

k = ops.rand(t, b*c)
v = ops.rand(t, b*c)
u = ops.rand(b*c)
w = ops.rand(b*c)
p = ops.rand(b*c)
q = ops.rand(b*c)
o = ops.rand(b*c)

# k = ops.randn(t)
# v = ops.randn(t)
# u = ops.randn(())
# w = ops.randn(())
# p = ops.randn(())
# q = ops.randn(())
# o = ops.randn(())
# print(type(o))
# print(k[0], o)
# print(k[0] + o)
# print(k.shape, u.shape)
# print(u.dtype)
# output = rwkv_time(k, v, w, u, p, q, o)

rwkv_map = ms.vmap(rwkv_time, (1, 1, 0, 0, 0, 0, 0))
output = rwkv_map(k, v, w, u, p, q, o)

print(output.shape)
