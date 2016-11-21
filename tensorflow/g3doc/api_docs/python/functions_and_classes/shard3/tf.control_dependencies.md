### `tf.control_dependencies(control_inputs)` {#control_dependencies}

Wrapper for `Graph.control_dependencies()` using the default graph.

See [`Graph.control_dependencies()`](../../api_docs/python/framework.md#Graph.control_dependencies)
for more details.

##### Args:


*  <b>`control_inputs`</b>: A list of `Operation` or `Tensor` objects which
    must be executed or computed before running the operations
    defined in the context.  Can also be `None` to clear the control
    dependencies.

##### Returns:

 A context manager that specifies control dependencies for all
 operations constructed within the context.

