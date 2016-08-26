### `tf.contrib.graph_editor.graph_replace(target_ts, replacement_ts, dst_scope='', src_scope='', reuse_dst_scope=False)` {#graph_replace}

Create a new graph which compute the targets from the replaced Tensors.

##### Args:


*  <b>`target_ts`</b>: a single tf.Tensor or an iterabble of tf.Tensor.
*  <b>`replacement_ts`</b>: dictionary mapping from original tensors to replaced tensors
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.
*  <b>`reuse_dst_scope`</b>: if True the dst_scope is re-used if it already exists.
    Otherwise, the scope is given a unique name based on the one given
    by appending an underscore followed by a digit (default).

##### Returns:

  A single tf.Tensor or a list of target tf.Tensor, depending on
  the type of the input argument `target_ts`.
  The returned tensors are recomputed using the tensors from replacement_ts.

##### Raises:


*  <b>`ValueError`</b>: if the targets are not connected to replacement_ts.

