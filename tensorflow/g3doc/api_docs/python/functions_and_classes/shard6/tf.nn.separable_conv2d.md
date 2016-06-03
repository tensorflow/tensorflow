### `tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None)` {#separable_conv2d}

2-D convolution with separable filters.

Performs a depthwise convolution that acts separately on channels followed by
a pointwise convolution that mixes channels.  Note that this is separability
between dimensions `[1, 2]` and `3`, not spatial separability between
dimensions `1` and `2`.

In detail,

    output[b, i, j, k] = sum_{di, dj, q, r]
        input[b, strides[1] * i + di, strides[2] * j + dj, q] *
        depthwise_filter[di, dj, q, r] *
        pointwise_filter[0, 0, q * channel_multiplier + r, k]

`strides` controls the strides for the depthwise convolution only, since
the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
`strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

##### Args:


*  <b>`input`</b>: 4-D `Tensor` with shape `[batch, in_height, in_width, in_channels]`.
*  <b>`depthwise_filter`</b>: 4-D `Tensor` with shape
    `[filter_height, filter_width, in_channels, channel_multiplier]`.
    Contains `in_channels` convolutional filters of depth 1.
*  <b>`pointwise_filter`</b>: 4-D `Tensor` with shape
    `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
    filter to mix channels after `depthwise_filter` has convolved spatially.
*  <b>`strides`</b>: 1-D of size 4.  The strides for the depthwise convolution for
    each dimension of `input`.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
    See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A 4-D `Tensor` of shape `[batch, out_height, out_width, out_channels]`.

##### Raises:


*  <b>`ValueError`</b>: If channel_multiplier * in_channels > out_channels,
    which means that the separable convolution is overparameterized.

