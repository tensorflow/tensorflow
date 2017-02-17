### `tf.contrib.graph_editor.compute_boundary_ts(ops)` {#compute_boundary_ts}

Compute the tensors at the boundary of a set of ops.

This function looks at all the tensors connected to the given ops (in/out)
and classify them into three categories:
1) input tensors: tensors whose generating operation is not in ops.
2) output tensors: tensors whose consumer operations are not in ops
3) inside tensors: tensors which are neither input nor output tensors.

Note that a tensor can be both an inside tensor and an output tensor if it is
consumed by operations both outside and inside of `ops`.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.

##### Returns:

  A tuple `(outside_input_ts, outside_output_ts, inside_ts)` where:
    `outside_input_ts` is a Python list of input tensors;
    `outside_output_ts` is a python list of output tensors;
    `inside_ts` is a python list of inside tensors.
  Since a tensor can be both an inside tensor and an output tensor,
  `outside_output_ts` and `inside_ts` might intersect.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.

