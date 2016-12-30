### `tf.contrib.graph_editor.get_consuming_ops(ts)` {#get_consuming_ops}

Return all the consuming ops of the tensors in ts.

##### Args:


*  <b>`ts`</b>: a list of `tf.Tensor`

##### Returns:

  A list of all the consuming `tf.Operation` of the tensors in `ts`.

##### Raises:


*  <b>`TypeError`</b>: if ts cannot be converted to a list of `tf.Tensor`.

