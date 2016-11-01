### `tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)` {#setdiff1d}

Returns the difference between the `x` and `y` treated as sets.

##### Args:


*  <b>`x`</b>: Set of values not assumed to be unique.
*  <b>`y`</b>: Set of values not assumed to be unique.
*  <b>`index_dtype`</b>: Output index type (`tf.int32`, `tf.int64`) default: `tf.int32`
*  <b>`name`</b>: A name for the operation (optional).


##### Returns:

  A `Tensor` the same type as `x` and `y`
  A `Tensor` that is of type `index_dtype` representing indices from .

