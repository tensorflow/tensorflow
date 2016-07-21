### `tf.nn.avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None)` {#avg_pool}

Performs the average pooling on the input.

Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.

##### Args:


*  <b>`value`</b>: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
    `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
*  <b>`ksize`</b>: A list of ints that has length >= 4.
    The size of the window for each dimension of the input tensor.
*  <b>`strides`</b>: A list of ints that has length >= 4.
    The stride of the sliding window for each dimension of the
    input tensor.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
*  <b>`data_format`</b>: A string. 'NHWC' and 'NCHW' are supported.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  A `Tensor` with the same type as `value`.  The average pooled output tensor.

