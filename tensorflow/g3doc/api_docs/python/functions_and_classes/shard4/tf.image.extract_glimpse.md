### `tf.image.extract_glimpse(input, size, offsets, centered=None, normalized=None, uniform_noise=None, name=None)` {#extract_glimpse}

Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location
`offsets` from the input tensor. If the windows only partially
overlaps the inputs, the non overlapping areas will be filled with
random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the
same as that of the input tensor. The height and width of the output
windows are specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`.
*  <b>`size`</b>: A `Tensor` of type `int32`.
*  <b>`offsets`</b>: A `Tensor` of type `float32`.
*  <b>`centered`</b>: An optional `bool`. Defaults to `True`.
*  <b>`normalized`</b>: An optional `bool`. Defaults to `True`.
*  <b>`uniform_noise`</b>: An optional `bool`. Defaults to `True`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.

