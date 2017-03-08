### `tf.contrib.graph_editor.remove_control_inputs(op, cops)` {#remove_control_inputs}

Remove the control inputs cops from co.

Warning: this function is directly manipulating the internals of the
`tf.Graph`.

##### Args:


*  <b>`op`</b>: a `tf.Operation` from which to remove the control inputs.
*  <b>`cops`</b>: an object convertible to a list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if op is not a `tf.Operation`.
*  <b>`ValueError`</b>: if any cop in cops is not a control input of op.

