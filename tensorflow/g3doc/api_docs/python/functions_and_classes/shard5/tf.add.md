### `tf.add(x, y, name=None)` {#add}

Returns x + y element-wise.

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.

