### `tf.contrib.graph_editor.copy_op_handler(info, op, copy_shape=True)` {#copy_op_handler}

Copy a `tf.Operation`.

##### Args:


*  <b>`info`</b>: Transform._TmpInfo instance.
*  <b>`op`</b>: the `tf.Operation` to be copied.
*  <b>`copy_shape`</b>: also copy the shape of the tensor

##### Returns:

  A `(op, op_outputs)` tuple containgin the transformed op and its outputs.

