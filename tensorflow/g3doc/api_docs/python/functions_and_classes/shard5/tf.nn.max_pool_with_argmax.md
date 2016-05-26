### `tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)` {#max_pool_with_argmax}

Performs max pooling on the input and outputs both max values and indices.

The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index
`((b * height + y) * width + x) * channels + c`.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`.
    4-D with shape `[batch, height, width, channels]`.  Input to pool over.
*  <b>`ksize`</b>: A list of `ints` that has length `>= 4`.
    The size of the window for each dimension of the input tensor.
*  <b>`strides`</b>: A list of `ints` that has length `>= 4`.
    The stride of the sliding window for each dimension of the
    input tensor.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`Targmax`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, argmax).

*  <b>`output`</b>: A `Tensor` of type `float32`. The max pooled output tensor.
*  <b>`argmax`</b>: A `Tensor` of type `Targmax`. 4-D.  The flattened indices of the max values chosen for each output.

