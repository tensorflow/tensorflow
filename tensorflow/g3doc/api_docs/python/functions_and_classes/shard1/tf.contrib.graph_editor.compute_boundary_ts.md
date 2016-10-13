### `tf.contrib.graph_editor.compute_boundary_ts(ops, ambiguous_ts_are_outputs=True)` {#compute_boundary_ts}

Compute the tensors at the boundary of a set of ops.

This function looks at all the tensors connected to the given ops (in/out)
and classify them into three categories:
1) input tensors: tensors whose generating operation is not in ops.
2) output tensors: tensors whose consumer operations are not in ops
3) inside tensors: tensors which are neither input nor output tensors.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.
*  <b>`ambiguous_ts_are_outputs`</b>: a tensor can have consumers both inside and
    outside ops. Such tensors are treated as outside tensor if
    ambiguous_ts_are_outputs is True, otherwise they are treated as
    inside tensor.

##### Returns:

  A tuple `(outside_input_ts, outside_output_ts, inside_ts)` where:
    `outside_input_ts` is a Python list of input tensors;
    `outside_output_ts` is a python list of output tensors;
    `inside_ts` is a python list of inside tensors.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.

