### `tf.contrib.framework.get_variable_full_name(var)` {#get_variable_full_name}

Returns the full name of a variable.

For normal Variables, this is the same as the var.op.name.  For
sliced or PartitionedVariables, this name is the same for all the
slices/partitions. In both cases, this is normally the name used in
a checkpoint file.

##### Args:


*  <b>`var`</b>: A `Variable` object.

##### Returns:

  A string that is the full name.

