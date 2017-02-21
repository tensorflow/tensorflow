### `tf.nn.conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)` {#conv2d_backprop_input}

Computes the gradients of convolution with respect to the input.

##### Args:


*  <b>`input_sizes`</b>: A `Tensor` of type `int32`.
    An integer vector representing the shape of `input`,
    where `input` is a 4-D `[batch, height, width, channels]` tensor.
*  <b>`filter`</b>: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    4-D with shape
    `[filter_height, filter_width, in_channels, out_channels]`.
*  <b>`out_backprop`</b>: A `Tensor`. Must have the same type as `filter`.
    4-D with shape `[batch, out_height, out_width, out_channels]`.
    Gradients w.r.t. the output of the convolution.
*  <b>`strides`</b>: A list of `ints`.
    The stride of the sliding window for each dimension of the input
    of the convolution. Must be in the same order as the dimension specified with
    format.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`use_cudnn_on_gpu`</b>: An optional `bool`. Defaults to `True`.
*  <b>`data_format`</b>: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
    Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `filter`.
  4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
  w.r.t. the input of the convolution.

