### `tf.contrib.graph_editor.get_forward_walk_ops(seed_ops, inclusive=True, within_ops=None, stop_at_ts=(), control_outputs=None)` {#get_forward_walk_ops}

Do a forward graph walk and return all the visited ops.

##### Args:


*  <b>`seed_ops`</b>: an iterable of operations from which the forward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the consumers of those tensors.
*  <b>`inclusive`</b>: if True the given seed_ops are also part of the resulting set.
*  <b>`within_ops`</b>: an iterable of `tf.Operation` within which the search is
    restricted. If `within_ops` is `None`, the search is performed within
    the whole graph.
*  <b>`stop_at_ts`</b>: an iterable of tensors at which the graph walk stops.
*  <b>`control_outputs`</b>: a `util.ControlOutputs` instance or None.
    If not `None`, it will be used while walking the graph forward.

##### Returns:

  A Python set of all the `tf.Operation` ahead of `seed_ops`.

##### Raises:


*  <b>`TypeError`</b>: if `seed_ops` or `within_ops` cannot be converted to a list of
    `tf.Operation`.

