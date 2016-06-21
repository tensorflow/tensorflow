### `tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)` {#depthwise_conv2d}

Depthwise 2-D convolution.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`
containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
applies a different filter to each input channel (expanding from 1 channel
to `channel_multiplier` channels for each), then concatenates the results
together.  The output has `in_channels * channel_multiplier` channels.

In detail,

    output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                     filter[di, dj, k, q]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the
same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

##### Args:


*  <b>`input`</b>: 4-D with shape `[batch, in_height, in_width, in_channels]`.
*  <b>`filter`</b>: 4-D with shape
    `[filter_height, filter_width, in_channels, channel_multiplier]`.
*  <b>`strides`</b>: 1-D of size 4.  The stride of the sliding window for each
    dimension of `input`.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
    See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A 4-D `Tensor` of shape
  `[batch, out_height, out_width, in_channels * channel_multiplier].`

