### `tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)` {#clip_by_value}

Clips tensor values to a specified min and max.

Given a tensor `t`, this operation returns a tensor of the same type and
shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
Any values less than `clip_value_min` are set to `clip_value_min`. Any values
greater than `clip_value_max` are set to `clip_value_max`.

##### Args:


*  <b>`t`</b>: A `Tensor`.
*  <b>`clip_value_min`</b>: A 0-D (scalar) `Tensor`. The minimum value to clip by.
*  <b>`clip_value_max`</b>: A 0-D (scalar) `Tensor`. The maximum value to clip by.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A clipped `Tensor`.

