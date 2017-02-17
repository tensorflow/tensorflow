### `tf.abs(x, name=None)` {#abs}

Computes the absolute value of a tensor.

Given a tensor of real numbers `x`, this operation returns a tensor
containing the absolute value of each element in `x`. For example, if x is
an input element and y is an output element, this operation computes
\\(y = |x|\\).

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor` of type `float32`, `float64`, `int32`, or
    `int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` the same size and type as `x` with absolute
    values.

