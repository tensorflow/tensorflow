### `tf.contrib.graph_editor.copy_with_input_replacements(sgv, replacement_ts, dst_graph=None, dst_scope='', src_scope='', reuse_dst_scope=False)` {#copy_with_input_replacements}

Copy a subgraph, replacing some of its inputs.

Note a replacement only happens if the tensor to be replaced
is an input of the given subgraph. The inputs of a subgraph can
be queried using sgv.inputs.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view. This argument is converted to a subgraph
    using the same rules as the function subgraph.make_view.
*  <b>`replacement_ts`</b>: dictionary mapping from original tensors to the
    replaced one.
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


*  <b>`TypeError`</b>: if dst_graph is not a tf.Graph.
*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules as the function subgraph.make_view.

