### `tf.fake_quant_with_min_max_vars_per_channel(inputs, min, max, name=None)` {#fake_quant_with_min_max_vars_per_channel}

Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,

`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
to 'outputs' tensor of same shape as `inputs`.

[min; max] is the clamping range for the 'inputs' data in the corresponding
depth channel.  Op divides this range into 255 steps (total of 256 values), then
replaces each 'inputs' value with the closest of the quantized step values.

This operation has a gradient and thus allows for training `min` and `max` values.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `float32`.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.

