### `tf.nn.conv3d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)` {#conv3d_transpose}

The transpose of `conv3d`.

This operation is sometimes called "deconvolution" after [Deconvolutional
Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
actually the transpose (gradient) of `conv3d` rather than an actual
deconvolution.

##### Args:


*  <b>`value`</b>: A 5-D `Tensor` of type `float` and shape
    `[batch, depth, height, width, in_channels]`.
*  <b>`filter`</b>: A 5-D `Tensor` with the same type as `value` and shape
    `[depth, height, width, output_channels, in_channels]`.  `filter`'s
    `in_channels` dimension must match that of `value`.
*  <b>`output_shape`</b>: A 1-D `Tensor` representing the output shape of the
    deconvolution op.
*  <b>`strides`</b>: A list of ints. The stride of the sliding window for each
    dimension of the input tensor.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
*  <b>`name`</b>: Optional name for the returned tensor.

##### Returns:

  A `Tensor` with the same type as `value`.

##### Raises:


*  <b>`ValueError`</b>: If input/output depth does not match `filter`'s shape, or if
    padding is other than `'VALID'` or `'SAME'`.

