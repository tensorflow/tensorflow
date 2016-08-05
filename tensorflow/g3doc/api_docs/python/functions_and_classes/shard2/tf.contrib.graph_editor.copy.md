### `tf.contrib.graph_editor.copy(sgv, dst_graph=None, dst_scope='', src_scope='')` {#copy}

Copy a subgraph.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view. This argument is converted to a subgraph
    using the same rules than the function subgraph.make_view.
*  <b>`dst_graph`</b>: the destination graph.
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.

##### Returns:

  the subgraph view of the copied subgraph.

##### Raises:


*  <b>`TypeError`</b>: if dst_graph is not a tf.Graph.
*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.

