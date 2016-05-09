### `tf.name_scope(name)` {#name_scope}

Wrapper for `Graph.name_scope()` using the default graph.

See
[`Graph.name_scope()`](../../api_docs/python/framework.md#Graph.name_scope)
for more details.

##### Args:


*  <b>`name`</b>: A name for the scope.

##### Returns:

  A context manager that installs `name` as a new name scope in the
  default graph.

