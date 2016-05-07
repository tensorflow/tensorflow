### `tf.contrib.layers.sum_regularizer(regularizer_list)` {#sum_regularizer}

Returns a function that applies the sum of multiple regularizers.

##### Args:


*  <b>`regularizer_list`</b>: A list of regularizers to apply.

##### Returns:

  A function with signature `sum_reg(weights, name=None)` that applies the
  sum of all the input regularizers.

