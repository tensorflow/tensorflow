### `tf.batch_matmul(x, y, adj_x=None, adj_y=None, name=None)` {#batch_matmul}

Multiplies slices of two tensors in batches.

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y

It is computed as:

    output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
    3-D or higher with shape `[..., r_x, c_x]`.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`.
    3-D or higher with shape `[..., r_y, c_y]`.
*  <b>`adj_x`</b>: An optional `bool`. Defaults to `False`.
    If `True`, adjoint the slices of `x`. Defaults to `False`.
*  <b>`adj_y`</b>: An optional `bool`. Defaults to `False`.
    If `True`, adjoint the slices of `y`. Defaults to `False`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.
  3-D or higher with shape `[..., r_o, c_o]`

