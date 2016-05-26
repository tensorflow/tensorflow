### `tf.pow(x, y, name=None)` {#pow}

Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```

##### Args:


*  <b>`x`</b>: A `Tensor` of type `float`, `double`, `int32`, `complex64`, or `int64`.
*  <b>`y`</b>: A `Tensor` of type `float`, `double`, `int32`, `complex64`, or `int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`.

