### `tf.nn.depthwise_conv2d_native_backprop_input(input_sizes, filter, out_backprop, strides, padding, name=None)` {#depthwise_conv2d_native_backprop_input}

Computes the gradients of depthwise convolution with respect to the input.

##### Args:


*  <b>`input_sizes`</b>: A `Tensor` of type `int32`.
    An integer vector representing the shape of `input`,
    where `input` is a 4-D `[batch, height, width, channels]` tensor.
*  <b>`filter`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    4-D with shape
    `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
*  <b>`out_backprop`</b>: A `Tensor`. Must have the same type as `filter`.
    4-D with shape `[batch, out_height, out_width, out_channels]`.
    Gradients w.r.t. the output of the convolution.
*  <b>`strides`</b>: A list of `ints`.
    The stride of the sliding window for each dimension of the input
    of the convolution.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `filter`.
  4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
  w.r.t. the input of the convolution.

