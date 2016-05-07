### `tf.saturate_cast(value, dtype, name=None)` {#saturate_cast}

Performs a safe saturating cast of `value` to `dtype`.

This function casts the input to `dtype` without applying any scaling.  If
there is a danger that values would over or underflow in the cast, this op
applies the appropriate clamping before the cast.

##### Args:


*  <b>`value`</b>: A `Tensor`.
*  <b>`dtype`</b>: The desired output `DType`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `value` safely cast to `dtype`.

