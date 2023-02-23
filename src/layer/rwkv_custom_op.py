
import numpy as np
import mindspore as ms
from mindspore.ops import ms_kernel

@ms_kernel
def rwkv_cell(k, v, frac_n, frac_d, scale, w_u, w_w):
    output = output_tensor(k.shape, k.dtype)
    next_frac_n = output_tensor(frac_n.shape, frac_n.dtype)
    next_frac_d = output_tensor(frac_d.shape, frac_d.dtype)
    next_scale = output_tensor(scale.shape, scale.dtype)
    softmax_scale = allocate(scale.shape, scale.dtype)

    for b in parallel(k.shape[0]):
        for c in vectorize(k.shape[1]):
            softmax_scale[b, c] = scale[b, c] if scale[b, c] > w_u[c] + k[b, c] else w_u[c] + k[b, c]
            output[b, c] = (exp(scale[b, c] - softmax_scale[b, c]) * frac_n[b, c] + exp(w_u[c] + k[b, c] - softmax_scale[b, c]) * v[b, c]) / \
                (exp(scale[b, c] - softmax_scale[b, c]) * frac_d[b, c] + exp(w_u[c] + k[b, c] - softmax_scale[b, c]))
            next_scale[b, c] = scale[b, c] + w_w[c] if scale[b, c] + w_w[c] > k[b, c] else k[b, c]
            next_frac_n[b, c] = exp(scale[b, c] + w_w[c] - next_scale[b, c]) * frac_n[b, c] + exp(k[b, c] - next_scale[b, c]) * v[b, c]
            next_frac_d[b, c] = exp(scale[b, c] + w_w[c] - next_scale[b, c]) * frac_d[b, c] + exp(k[b, c] - next_scale[b, c])

    return output, next_frac_n, next_frac_d, next_scale

@ms_kernel
def rwkv_forward(k, v, w, u):
    y = output_tensor(k.shape, k.dtype)

    for b in parallel(k.shape[1]):
        for c in parallel(k.shape[2]):
            p = float32(0)
            q = float32(0)
            o = float32(-1e38)
            for t in serial(k.shape[0]):
                no = o if o > u[c] + k[t, b, c] else u[c] + k[t, b, c]
                a = exp(o - no)
                bb = exp(u[c] + k[t, b, c] - no)
                y[t, b, c] = (a * p + bb * v[t, b, c]) / (a * q + bb)

                no = w[c] + o if w[c] + o > k[t, b, c] else k[t, b, c]
                a = exp(w[c] + o - no)
                bb = exp(k[t, b, c] - no)
                p = a * p + bb * v[t, b, c]
                q = a * q + bb
                o = no
    return y

@ms_kernel
def rwkv_backward(k, v, w, u, gy):
    _gw = output_tensor(k.shape[1:], w.dtype)
    _gu = output_tensor(k.shape[1:], u.dtype)
    _gk = output_tensor(k.shape, k.dtype)
    _gv = output_tensor(v.shape, v.dtype)

    for b in parallel(k.shape[1]):
        for c in parallel(k.shape[2]):
            y = allocate((k.shape[0],), k.dtype)
            z = allocate((k.shape[0],), k.dtype)
            zexp = allocate((k.shape[0],), k.dtype)
            gw = float32(0)
            gu = float32(0)
            q = float32(0)
            p = float32(0)
            o = float32(-1e38)
            dpdw = float32(0)
            dqdw = float32(0)
            for t in serial(k.shape[0]):
                no = o if o > u[c] + k[t, b, c] else u[c] + k[t, b, c]
                a = exp(o - no)
                bb = exp(u[c] + k[t, b, c] - no)

                num = a * p + bb * v[t, b, c]
                iden = 1 / (a * q + bb)
                y[t] = num * iden
                z[t] = iden
                zexp[t] = u[c] + k[t, b, c] - no

                gw += gy[t, b, c] * (dpdw - dqdw * y[t]) * iden * a
                gu += gy[t, b, c] * (v[t, b, c] - y[t]) * bb * iden

                no = w[c] + o if w[c] + o > k[t, b, c] else k[t, b, c]
                a = exp(w[c] + o - no)
                bb = exp(k[t, b, c] - no)
                dpdw = a * (p + dpdw)
                dqdw = a * (q + dqdw)
                p = a * p + bb * v[t, b, c]
                q = a * q + bb
                o = no
            
            gq = float32(0)
            gp = float32(0)
            oo = float32(-1e38)
            max_t = int32(k.shape[0] - 1)
            for t in serial(k.shape[0]):
                aa = gy[max_t - t, b, c] * z[max_t - t] * exp(zexp[max_t - t])
                bbb = exp(k[max_t - t, b, c] + oo)
                _gk[max_t - t, b, c] = aa * (v[max_t - t, b, c] * y[max_t - t]) + bbb * (gp * v[max_t - t, b, c] + gq)
                _gv[max_t - t, b, c] = aa + bbb * gp

                noo = w[c] + oo if w[c] + oo > zexp[max_t - t] - k[max_t - t, b, c] - u[c] else zexp[max_t - t] - k[max_t - t, b, c] - u[c]
                aa = exp(w[c] + oo - noo)
                bbb = gy[max_t - t, b, c] * z[max_t - t] * exp(zexp[max_t - t] - k[max_t - t, b, c] - u[c] - noo)
                gp = aa * gp + bbb
                gq = aa * gq - bbb * y[max_t - t]
                oo = noo

            _gw[b, c] += gw * w[c]
            _gu[b, c] += gu
    return _gk, _gv, _gw, _gu