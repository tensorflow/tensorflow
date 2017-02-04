### `tf_debug.watch_graph(run_options, graph, debug_ops='DebugIdentity', debug_urls=None, node_name_regex_whitelist=None, op_type_regex_whitelist=None)` {#watch_graph}

Add debug watches to `RunOptions` for a TensorFlow graph.

To watch all `Tensor`s on the graph, let both `node_name_regex_whitelist`
and `op_type_regex_whitelist` be the default (`None`).

N.B.: Under certain circumstances, not all specified `Tensor`s will be
  actually watched (e.g., nodes that are constant-folded during runtime will
  not be watched).

##### Args:


*  <b>`run_options`</b>: An instance of `config_pb2.RunOptions` to be modified.
*  <b>`graph`</b>: An instance of `ops.Graph`.
*  <b>`debug_ops`</b>: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
*  <b>`debug_urls`</b>: URLs to send debug values to. Can be a list of strings,
    a single string, or None. The case of a single string is equivalent to
    a list consisting of a single string, e.g., `file:///tmp/tfdbg_dump_1`,
    `grpc://localhost:12345`.
*  <b>`node_name_regex_whitelist`</b>: Regular-expression whitelist for node_name,
    e.g., `"(weight_[0-9]+|bias_.*)"`
*  <b>`op_type_regex_whitelist`</b>: Regular-expression whitelist for the op type of
    nodes, e.g., `"(Variable|Add)"`.
    If both `node_name_regex_whitelist` and `op_type_regex_whitelist`
    are set, the two filtering operations will occur in a logical `AND`
    relation. In other words, a node will be included if and only if it
    hits both whitelists.

