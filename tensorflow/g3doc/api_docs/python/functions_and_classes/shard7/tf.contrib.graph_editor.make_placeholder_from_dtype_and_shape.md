### `tf.contrib.graph_editor.make_placeholder_from_dtype_and_shape(dtype, shape=None, scope=None)` {#make_placeholder_from_dtype_and_shape}

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

