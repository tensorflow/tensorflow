### `tf.contrib.graph_editor.op_type(op_types, op=None)` {#op_type}

Check if an op is of the given type.

##### Args:


*  <b>`op_types`</b>: tuple of strings containing the types to check against.
    For instance: ("Add", "Const")
*  <b>`op`</b>: the operation to check (or None).

##### Returns:

  if op is not None, return True if the op is of the correct type.
  if op is None, return a lambda function which does the type checking.

