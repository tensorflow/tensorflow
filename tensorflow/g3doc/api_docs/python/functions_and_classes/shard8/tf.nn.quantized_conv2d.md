### `tf.nn.quantized_conv2d(input, filter, min_input, max_input, min_filter, max_filter, strides, padding, out_type=None, name=None)` {#quantized_conv2d}

Computes a 2D convolution given quantized 4D input and filter tensors.

The inputs are quantized tensors where the lowest value represents the real
number of the associated minimum, and the highest represents the maximum.
This means that you can only interpret the quantized output in the same way, by
taking the returned minimum and maximum values into account.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
*  <b>`filter`</b>: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    filter's input_depth dimension must match input's depth dimensions.
*  <b>`min_input`</b>: A `Tensor` of type `float32`.
    The float value that the lowest quantized input value represents.
*  <b>`max_input`</b>: A `Tensor` of type `float32`.
    The float value that the highest quantized input value represents.
*  <b>`min_filter`</b>: A `Tensor` of type `float32`.
    The float value that the lowest quantized filter value represents.
*  <b>`max_filter`</b>: A `Tensor` of type `float32`.
    The float value that the highest quantized filter value represents.
*  <b>`strides`</b>: A list of `ints`.
    The stride of the sliding window for each dimension of the input
    tensor.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`out_type`</b>: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.qint32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, min_output, max_output).

*  <b>`output`</b>: A `Tensor` of type `out_type`.
*  <b>`min_output`</b>: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
*  <b>`max_output`</b>: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.

