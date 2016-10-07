### `tf.nn.conv1d(value, filters, stride, padding, use_cudnn_on_gpu=None, data_format=None, name=None)` {#conv1d}

Computes a 1-D convolution given 3-D input and filter tensors.

Given an input tensor of shape [batch, in_width, in_channels]
and a filter / kernel tensor of shape
[filter_width, in_channels, out_channels], this op reshapes
the arguments to pass them to conv2d to perform the equivalent
convolution operation.

Internally, this op reshapes the input tensors and invokes
`tf.nn.conv2d`.  A tensor of shape [batch, in_width, in_channels]
is reshaped to [batch, 1, in_width, in_channels], and the filter
is reshaped to [1, filter_width, in_channels, out_channels].
The result is then reshaped back to [batch, out_width, out_channels]
(where out_width is a function of the stride and padding as in
conv2d) and returned to the caller.

##### Args:


*  <b>`value`</b>: A 3D `Tensor`.  Must be of type `float32` or `float64`.
*  <b>`filters`</b>: A 3D `Tensor`.  Must have the same type as `input`.
*  <b>`stride`</b>: An `integer`.  The number of entries by which
    the filter is moved right at each step.
*  <b>`padding`</b>: 'SAME' or 'VALID'
*  <b>`use_cudnn_on_gpu`</b>: An optional `bool`.  Defaults to `True`.
*  <b>`data_format`</b>: An optional `string` from `"NHWC", "NCHW"`.  Defaults
    to `"NHWC"`, the data is stored in the order of
    [batch, in_width, in_channels].  The `"NCHW"` format stores
    data as [batch, in_channels, in_width].
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`.  Has the same type as input.

