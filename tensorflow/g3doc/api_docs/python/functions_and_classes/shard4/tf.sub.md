### `tf.sub(*args, **kwargs)` {#sub}

Returns x - y element-wise.

  *NOTE*: `Sub` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
`tf.sub(x, y)` is deprecated, please use `tf.subtract(x, y)` or `x - y`

