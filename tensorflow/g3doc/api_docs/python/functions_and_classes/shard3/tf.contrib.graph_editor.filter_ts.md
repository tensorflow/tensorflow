### `tf.contrib.graph_editor.filter_ts(ops, positive_filter)` {#filter_ts}

Get all the tensors which are input or output of an op in ops.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.
*  <b>`positive_filter`</b>: a function deciding whether to keep a tensor or not.
    If True, all the tensors are returned.

##### Returns:

  A list of tf.Tensor.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.

