### `tf.contrib.layers.unit_norm(*args, **kwargs)` {#unit_norm}

Normalizes the given input across the specified dimension to unit length.

Note that the rank of `input` must be known.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of arbitrary size.
*  <b>`dim`</b>: The dimension along which the input is normalized.
*  <b>`epsilon`</b>: A small value to add to the inputs to avoid dividing by zero.
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  The normalized `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: If dim is smaller than the number of dimensions in 'inputs'.

