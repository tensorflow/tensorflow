### `tf.contrib.graph_editor.transform_op_in_place(info, op, detach_outputs=False)` {#transform_op_in_place}

Transform a op in-place - experimental!

Transform an operation in place. It reconnects the inputs if they have been
modified. if detach_outputs is True, the outputs of op are also detached.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`op`</b>: the op to transform in place.
*  <b>`detach_outputs`</b>: if True, the outputs of op are detached, ready for the user
    to add more operation.

##### Returns:

  The transformed op.

