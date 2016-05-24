### `tf.device(device_name_or_function)` {#device}

Wrapper for `Graph.device()` using the default graph.

See
[`Graph.device()`](../../api_docs/python/framework.md#Graph.device)
for more details.

##### Args:


*  <b>`device_name_or_function`</b>: The device name or function to use in
    the context.

##### Returns:

  A context manager that specifies the default device to use for newly
  created ops.

