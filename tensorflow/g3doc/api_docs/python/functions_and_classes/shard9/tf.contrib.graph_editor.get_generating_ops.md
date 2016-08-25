### `tf.contrib.graph_editor.get_generating_ops(ts)` {#get_generating_ops}

Return all the generating ops of the tensors in ts.

##### Args:


*  <b>`ts`</b>: a list of tf.Tensor

##### Returns:

  A list of all the generating tf.Operation of the tensors in ts.

##### Raises:


*  <b>`TypeError`</b>: if ts cannot be converted to a list of tf.Tensor.

