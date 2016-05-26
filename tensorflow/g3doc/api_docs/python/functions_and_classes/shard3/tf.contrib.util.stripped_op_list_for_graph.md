### `tf.contrib.util.stripped_op_list_for_graph(graph_def)` {#stripped_op_list_for_graph}

Collect the stripped OpDefs for ops used by a graph.

This function computes the `stripped_op_list` field of `MetaGraphDef` and
similar protos.  The result can be communicated from the producer to the
consumer, which can then use the C++ function
`RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### Returns:

  An `OpList` of ops used by the graph.

##### Raises:


*  <b>`ValueError`</b>: If an unregistered op is used.

