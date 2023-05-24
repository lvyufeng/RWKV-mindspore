# RWKV-mindspore

## WKV operator

### WKV forward
- attributes:
  - B: batch_size
  - T: seq_length
  - C: hidden_size
- inputs:
    - time_first/w: (hidden_size,)
    - time_decay/u: (hidden_size,)
    - key: (batch_size, seq_length, hidden_size)
    - value: (batch_size, seq_length, hidden_size)
- output:
    - output: (batch_size, seq_length, hidden_size)

#### equation

![RWKV-v3-plan](https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v3-plan.png)

### WKV forward（huggingface）
- attributes:
  - B: batch_size
  - T: seq_length
  - C: hidden_size

- inputs:
  - time_first/w: (hidden_size,)
  - time_decay/u: (hidden_size,)
  - key: (batch_size, seq_length, hidden_size)
  - value: (batch_size, seq_length, hidden_size)
  - states: (batch_size, hidden_size, 3), for **aa, bb**, initial with zeros, **pp** initial with -1e-38.
- output:
  - output: (batch_size, seq_length, hidden_size)

### WKV backward

- attributes:
  - B: batch_size
  - T: seq_length
  - C: hidden_size

- inputs:
  - time_first/w: (hidden_size,)
  - time_decay/u: (hidden_size,)
  - key: (batch_size, seq_length, hidden_size)
  - value: (batch_size, seq_length, hidden_size)
  - gy: (batch_size, seq_length, hidden_size)
- outputs:
  - gw: (batch_size, hidden_size)
  - gu: (batch_size, hidden_size)
  - gk: (batch_size, seq_length, hidden_size)
  - gv: (batch_size, seq_length, hidden_size)

## Primitive(Cell)

```python
class WKV(nn.Cell):
    def construct(self, w, u, k, v):
        w = -ops.exp(w)
        y = wkv_forward_op(w, u, k, v)

        return y
    
    def bprop(self, w, u, k, v, y, gy):
        gw, gu, gk, gv = wkv_backward_op(w, u, k, v, gy)
        gw = ops.sum(gw, 1)
        gu = ops.sum(gu, 1)

        return (gw, gu, gk, gv)
```