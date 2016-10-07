### `tf.contrib.graph_editor.get_within_boundary_ops(ops, seed_ops, boundary_ops=(), inclusive=True, control_inputs=False, control_outputs=None, control_ios=None)` {#get_within_boundary_ops}

Return all the `tf.Operation` within the given boundary.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of `tf.Operation`. those ops define the
    set in which to perform the operation (if a `tf.Graph` is given, it
    will be converted to the list of all its operations).
*  <b>`seed_ops`</b>: the operations from which to start expanding.
*  <b>`boundary_ops`</b>: the ops forming the boundary.
*  <b>`inclusive`</b>: if `True`, the result will also include the boundary ops.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of `util.ControlOutputs` or `None`. If not
    `None`, control outputs are enabled.
*  <b>`control_ios`</b>: An instance of `util.ControlOutputs` or `None`. If not
    `None`, both control inputs and control outputs are enabled. This is
    equivalent to set control_inputs to True and control_outputs to
    the `util.ControlOutputs` instance.

##### Returns:

  All the `tf.Operation` surrounding the given ops.

##### Raises:


*  <b>`TypeError`</b>: if `ops` or `seed_ops` cannot be converted to a list of
    `tf.Operation`.
*  <b>`ValueError`</b>: if the boundary is intersecting with the seeds.

