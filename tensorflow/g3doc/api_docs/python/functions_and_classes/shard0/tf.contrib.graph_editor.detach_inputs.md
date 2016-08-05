### `tf.contrib.graph_editor.detach_inputs(sgv, control_inputs=False)` {#detach_inputs}

Detach the inputs of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
*  <b>`control_inputs`</b>: if True control_inputs are also detached.

##### Returns:

  A new subgraph view of the detached subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.

