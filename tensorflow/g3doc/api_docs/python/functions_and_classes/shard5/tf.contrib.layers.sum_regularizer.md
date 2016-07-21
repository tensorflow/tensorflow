### `tf.contrib.layers.sum_regularizer(regularizer_list, scope=None)` {#sum_regularizer}

Returns a function that applies the sum of multiple regularizers.

##### Args:


*  <b>`regularizer_list`</b>: A list of regularizers to apply.
*  <b>`scope`</b>: An optional op_scope name

##### Returns:

  A function with signature `sum_reg(weights)` that applies the
  sum of all the input regularizers.

