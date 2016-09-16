### `tf.contrib.graph_editor.make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False)` {#make_list_of_t}

Convert ts to a list of tf.Tensor.

##### Args:


*  <b>`ts`</b>: can be an iterable of tf.Tensor, a tf.Graph or a single tensor.
*  <b>`check_graph`</b>: if True check if all the tensors belong to the same graph.
*  <b>`allow_graph`</b>: if False a tf.Graph cannot be converted.
*  <b>`ignore_ops`</b>: if True, silently ignore tf.Operation.

##### Returns:

  A newly created list of tf.Tensor.

##### Raises:


*  <b>`TypeError`</b>: if ts cannot be converted to a list of tf.Tensor or,
   if check_graph is True, if all the ops do not belong to the same graph.

