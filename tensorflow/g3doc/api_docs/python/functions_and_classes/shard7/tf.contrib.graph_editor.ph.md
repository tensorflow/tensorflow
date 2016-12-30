### `tf.contrib.graph_editor.ph(dtype, shape=None, scope=None)` {#ph}

Create a tf.placeholder for the Graph Editor.

Note that the correct graph scope must be set by the calling function.
The placeholder is named using the function placeholder_name (with no
tensor argument).

##### Args:


*  <b>`dtype`</b>: the tensor type.
*  <b>`shape`</b>: the tensor shape (optional).
*  <b>`scope`</b>: absolute scope within which to create the placeholder. None
    means that the scope of t is preserved. "" means the root scope.

##### Returns:

  A newly created tf.placeholder.

