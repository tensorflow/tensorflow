### `tf.contrib.util.ops_used_by_graph_def(graph_def)` {#ops_used_by_graph_def}

Collect the list of ops used by a graph.

Does not validate that the ops are all registered.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### Returns:

  A list of strings, each naming an op used by the graph.

