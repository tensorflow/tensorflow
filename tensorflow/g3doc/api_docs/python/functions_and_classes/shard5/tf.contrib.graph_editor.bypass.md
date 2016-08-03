### `tf.contrib.graph_editor.bypass(sgv)` {#bypass}

Bypass the given subgraph by connecting its inputs to its outputs.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be bypassed. This argument is converted to a
    subgraph using the same rules than the function subgraph.make_view.

##### Returns:

  A new subgraph view of the bypassed subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.

