### `tf.sigmoid(x, name=None)` {#sigmoid}

Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.

##### Args:


*  <b>`x`</b>: A Tensor with type `float32`, `float64`, `int32`, `complex64`, `int64`,
    or `qint32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A Tensor with the same type as `x` if `x.dtype != qint32`
    otherwise the return type is `quint8`.

@compatibility(numpy)
Equivalent to np.scipy.special.expit
@end_compatibility

