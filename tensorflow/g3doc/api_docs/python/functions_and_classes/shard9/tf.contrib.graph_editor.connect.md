### `tf.contrib.graph_editor.connect(sgv0, sgv1, disconnect_first=False)` {#connect}

Connect the outputs of sgv0 to the inputs of sgv1.

##### Args:


*  <b>`sgv0`</b>: the first subgraph to have its outputs swapped. This argument is
    converted to a subgraph using the same rules as the function
    subgraph.make_view.
*  <b>`sgv1`</b>: the second subgraph to have its outputs swapped. This argument is
    converted to a subgraph using the same rules as the function
    subgraph.make_view.
*  <b>`disconnect_first`</b>: if True the current outputs of sgv0 are disconnected.

##### Returns:

  Two new subgraph views (now connected). sgv0 and svg1 are also modified
    in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv0 or sgv1 cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.

