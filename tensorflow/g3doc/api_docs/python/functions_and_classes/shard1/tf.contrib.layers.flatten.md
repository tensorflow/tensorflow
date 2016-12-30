### `tf.contrib.layers.flatten(*args, **kwargs)` {#flatten}

Flattens the input while maintaining the batch_size.

  Assumes that the first dimension represents the batch.

##### Args:


*  <b>`inputs`</b>: a tensor of size [batch_size, ...].
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  a flattened tensor with shape [batch_size, k].

##### Raises:


*  <b>`ValueError`</b>: if inputs.dense_shape is wrong.

