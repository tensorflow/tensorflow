### `tf_debug.add_debug_tensor_watch(run_options, node_name, output_slot=0, debug_ops='DebugIdentity', debug_urls=None)` {#add_debug_tensor_watch}

Add watch on a `Tensor` to `RunOptions`.

N.B.: Under certain circumstances, the `Tensor` may not be actually watched
  (e.g., if the node of the `Tensor` is constant-folded during runtime).

##### Args:


*  <b>`run_options`</b>: An instance of `config_pb2.RunOptions` to be modified.
*  <b>`node_name`</b>: (`str`) name of the node to watch.
*  <b>`output_slot`</b>: (`int`) output slot index of the tensor from the watched node.
*  <b>`debug_ops`</b>: (`str` or `list` of `str`) name(s) of the debug op(s). Can be a
    `list` of `str` or a single `str`. The latter case is equivalent to a
    `list` of `str` with only one element.
*  <b>`debug_urls`</b>: (`str` or `list` of `str`) URL(s) to send debug values to,
    e.g., `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.

