### `tf.group(*inputs, **kwargs)` {#group}

Create an op that groups multiple operations.

When this op finishes, all ops in `input` have finished. This op has no
output.

See also `tuple` and `with_dependencies`.

##### Args:


*  <b>`*inputs`</b>: Zero or more tensors to group.
*  <b>`**kwargs`</b>: Optional parameters to pass when constructing the NodeDef.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  An Operation that executes all its inputs.

##### Raises:


*  <b>`ValueError`</b>: If an unknown keyword argument is provided.

