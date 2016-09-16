### `tf.contrib.graph_editor.get_walks_intersection_ops(forward_seed_ops, backward_seed_ops, forward_inclusive=True, backward_inclusive=True, within_ops=None, control_inputs=False, control_outputs=None, control_ios=None)` {#get_walks_intersection_ops}

Return the intersection of a foward and a backward walk.

##### Args:


*  <b>`forward_seed_ops`</b>: an iterable of operations from which the forward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the consumers of those tensors.
*  <b>`backward_seed_ops`</b>: an iterable of operations from which the backward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the generators of those tensors.
*  <b>`forward_inclusive`</b>: if True the given forward_seed_ops are also part of the
    resulting set.
*  <b>`backward_inclusive`</b>: if True the given backward_seed_ops are also part of the
    resulting set.
*  <b>`within_ops`</b>: an iterable of tf.Operation whithin which the search is
    restricted. If within_ops is None, the search is performed within
    the whole graph.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A Python set of all the tf.Operation in the intersection of a foward and a
    backward walk.

##### Raises:


*  <b>`TypeError`</b>: if forward_seed_ops or backward_seed_ops or within_ops cannot be
    converted to a list of tf.Operation.

