### `tf.sign(x, name=None)` {#sign}

Returns an element-wise indication of the sign of a number.

`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.

