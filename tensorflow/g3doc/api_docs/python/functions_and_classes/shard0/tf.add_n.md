### `tf.add_n(inputs, name=None)` {#add_n}

Adds all input tensors element-wise.

##### Args:


*  <b>`inputs`</b>: A list of `Tensor` objects, each with same shape and type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of same shape and type as the elements of `inputs`.

##### Raises:


*  <b>`ValueError`</b>: If `inputs` don't all have same shape and dtype or the shape
  cannot be inferred.

