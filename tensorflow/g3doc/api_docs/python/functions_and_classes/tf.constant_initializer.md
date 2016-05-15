### `tf.constant_initializer(value=0.0, dtype=tf.float32)` {#constant_initializer}

Returns an initializer that generates tensors with a single value.

##### Args:


*  <b>`value`</b>: A Python scalar. All elements of the initialized variable
    will be set to this value.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a single value.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.

