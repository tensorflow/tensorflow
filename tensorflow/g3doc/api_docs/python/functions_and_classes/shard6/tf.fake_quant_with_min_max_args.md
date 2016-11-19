### `tf.fake_quant_with_min_max_args(inputs, min=None, max=None, name=None)` {#fake_quant_with_min_max_args}

Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

Attributes [min; max] define the clamping range for the 'inputs' data.  Op
divides this range into 255 steps (total of 256 values), then replaces each
'inputs' value with the closest of the quantized step values.

Quantization is called fake since the output is still in floating point.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `float32`.
*  <b>`min`</b>: An optional `float`. Defaults to `-6`.
*  <b>`max`</b>: An optional `float`. Defaults to `6`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.

