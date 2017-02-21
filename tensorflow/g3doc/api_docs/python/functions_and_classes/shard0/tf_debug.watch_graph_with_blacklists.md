### `tf_debug.watch_graph_with_blacklists(run_options, graph, debug_ops='DebugIdentity', debug_urls=None, node_name_regex_blacklist=None, op_type_regex_blacklist=None, global_step=-1)` {#watch_graph_with_blacklists}

Add debug tensor watches, blacklisting nodes and op types.

This is similar to `watch_graph()`, but the node names and op types are
blacklisted, instead of whitelisted.

N.B.: Under certain circumstances, not all specified `Tensor`s will be
  actually watched (e.g., nodes that are constant-folded during runtime will
  not be watched).

##### Args:


*  <b>`run_options`</b>: An instance of `config_pb2.RunOptions` to be modified.
*  <b>`graph`</b>: An instance of `ops.Graph`.
*  <b>`debug_ops`</b>: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
*  <b>`debug_urls`</b>: URL(s) to send ebug values to, e.g.,
    `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.
*  <b>`node_name_regex_blacklist`</b>: Regular-expression blacklist for node_name.
    This should be a string, e.g., `"(weight_[0-9]+|bias_.*)"`.
*  <b>`op_type_regex_blacklist`</b>: Regular-expression blacklist for the op type of
    nodes, e.g., `"(Variable|Add)"`.
    If both node_name_regex_blacklist and op_type_regex_blacklist
    are set, the two filtering operations will occur in a logical `OR`
    relation. In other words, a node will be excluded if it hits either of
    the two blacklists; a node will be included if and only if it hits
    neither of the blacklists.
*  <b>`global_step`</b>: (`int`) Optional global_step count for this debug tensor
    watch.

