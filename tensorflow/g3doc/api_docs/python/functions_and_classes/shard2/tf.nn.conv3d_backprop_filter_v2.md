### `tf.nn.conv3d_backprop_filter_v2(input, filter_sizes, out_backprop, strides, padding, name=None)` {#conv3d_backprop_filter_v2}

Computes the gradients of 3-D convolution with respect to the filter.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Shape `[batch, depth, rows, cols, in_channels]`.
*  <b>`filter_sizes`</b>: A `Tensor` of type `int32`.
    An integer vector representing the tensor shape of `filter`,
    where `filter` is a 5-D
    `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
    tensor.
*  <b>`out_backprop`</b>: A `Tensor`. Must have the same type as `input`.
    Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
    out_channels]`.
*  <b>`strides`</b>: A list of `ints` that has length `>= 5`.
    1-D tensor of length 5. The stride of the sliding window for each
    dimension of `input`. Must have `strides[0] = strides[4] = 1`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.

