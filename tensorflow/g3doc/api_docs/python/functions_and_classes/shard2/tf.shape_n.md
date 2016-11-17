### `tf.shape_n(input, out_type=None, name=None)` {#shape_n}

Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

##### Args:


*  <b>`input`</b>: A list of at least 1 `Output` objects of the same type.
*  <b>`out_type`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list with the same number of `Output` objects as `input` of `Output` objects of type out_type.

