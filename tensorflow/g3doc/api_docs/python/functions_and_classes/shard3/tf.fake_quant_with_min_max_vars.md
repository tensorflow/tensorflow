### `tf.fake_quant_with_min_max_vars(inputs, min, max, name=None)` {#fake_quant_with_min_max_vars}

Fake-quantize the 'inputs' tensor of type float via global float scalars `min`

and `max` to 'outputs' tensor of same shape as `inputs`.

[min; max] is the clamping range for the 'inputs' data.  Op divides this range
into 255 steps (total of 256 values), then replaces each 'inputs' value with the
closest of the quantized step values.

This operation has a gradient and thus allows for training `min` and `max` values.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `float32`.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.

