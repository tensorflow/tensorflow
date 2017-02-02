### `tf.contrib.graph_editor.copy(sgv, dst_graph=None, dst_scope='', src_scope='', reuse_dst_scope=False)` {#copy}

Copy a subgraph.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view. This argument is converted to a subgraph
    using the same rules than the function subgraph.make_view.
*  <b>`dst_graph`</b>: the destination graph.
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.
*  <b>`reuse_dst_scope`</b>: if True the dst_scope is re-used if it already exists.
    Otherwise, the scope is given a unique name based on the one given
    by appending an underscore followed by a digit (default).

##### Returns:

  A tuple `(sgv, info)` where:
    `sgv` is the transformed subgraph view;
    `info` is an instance of Transformer.ResultInfo containing
    information about the transform, including mapping between
    original and transformed tensors and operations.

##### Raises:


*  <b>`TypeError`</b>: if `dst_graph` is not a `tf.Graph`.
*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.

