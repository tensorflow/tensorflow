### `tf.trace(x, name=None)` {#trace}

Compute the trace of a tensor `x`.

`trace(x)` returns the sum along the main diagonal of each inner-most matrix
in x. If x is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output
is a tensor of rank `k-2` with dimensions `[I, J, K, ..., L]` where

`output[i, j, k, ..., l] = trace(x[i, j, i, ..., l, :, :])`

For example:

```python
# 'x' is [[1, 2],
#         [3, 4]]
tf.trace(x) ==> 5

# 'x' is [[1,2,3],
#         [4,5,6],
#         [7,8,9]]
tf.trace(x) ==> 15

# 'x' is [[[1,2,3],
#          [4,5,6],
#          [7,8,9]],
#         [[-1,-2,-3],
#          [-4,-5,-6],
#          [-7,-8,-9]]]
tf.trace(x) ==> [15,-15]
```

##### Args:


*  <b>`x`</b>: tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The trace of input tensor.

