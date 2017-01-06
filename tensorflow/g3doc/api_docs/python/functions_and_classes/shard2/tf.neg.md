### `tf.neg(*args, **kwargs)` {#neg}

Computes numerical negative value element-wise. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
`tf.neg(x)` is deprecated, please use `tf.negative(x)` or `-x`

I.e., \(y = -x\).

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
    `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

