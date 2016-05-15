### `tf.nn.max_pool3d(input, ksize, strides, padding, name=None)` {#max_pool3d}

Performs 3D max pooling on the input.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
*  <b>`ksize`</b>: A list of `ints` that has length `>= 5`.
    1-D tensor of length 5. The size of the window for each dimension of
    the input tensor. Must have `ksize[0] = ksize[1] = 1`.
*  <b>`strides`</b>: A list of `ints` that has length `>= 5`.
    1-D tensor of length 5. The stride of the sliding window for each
    dimension of `input`. Must have `strides[0] = strides[4] = 1`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`. The max pooled output tensor.

