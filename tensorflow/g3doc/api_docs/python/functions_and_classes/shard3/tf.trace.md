### `tf.trace(x, name=None)` {#trace}

Compute the trace of a tensor `x`.

`trace(x)` returns the sum of along the diagonal.

For example:

```python
# 'x' is [[1, 1],
#         [1, 1]]
tf.trace(x) ==> 2

# 'x' is [[1,2,3],
#         [4,5,6],
#         [7,8,9]]
tf.trace(x) ==> 15
```

##### Args:


*  <b>`x`</b>: 2-D tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The trace of input tensor.

