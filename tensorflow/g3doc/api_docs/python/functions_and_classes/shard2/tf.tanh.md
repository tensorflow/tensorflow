### `tf.tanh(x, name=None)` {#tanh}

Computes hyperbolic tangent of `x` element-wise.

##### Args:


*  <b>`x`</b>: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
    or `qint32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
    the return type is `quint8`.

