### `tf.abs(x, name=None)` {#abs}

Computes the absolute value of a tensor.

Given a tensor of real numbers `x`, this operation returns a tensor
containing the absolute value of each element in `x`. For example, if x is
an input element and y is an output element, this operation computes
\\(y = |x|\\).

See [`tf.complex_abs()`](#tf_complex_abs) to compute the absolute value of a complex
number.

##### Args:


*  <b>`x`</b>: A `Tensor` of type `float`, `double`, `int32`, or `int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

   A `Tensor` the same size and type as `x` with absolute values.

