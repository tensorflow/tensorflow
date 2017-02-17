### `tf.contrib.copy_graph.copy_op_to_graph(org_instance, to_graph, variables, scope='')` {#copy_op_to_graph}

Given an `Operation` 'org_instance` from one `Graph`,
initializes and returns a copy of it from another `Graph`,
under the specified scope (default `""`).

The copying is done recursively, so any `Operation` whose output
is required to evaluate the `org_instance`, is also copied (unless
already done).

Since `Variable` instances are copied separately, those required
to evaluate `org_instance` must be provided as input.

Args:
org_instance: An `Operation` from some `Graph`. Could be a
    `Placeholder` as well.
to_graph: The `Graph` to copy `org_instance` to.
variables: An iterable of `Variable` instances to copy `org_instance` to.
scope: A scope for the new `Variable` (default `""`).

##### Returns:

    The copied `Operation` from `to_graph`.

##### Raises:


*  <b>`TypeError`</b>: If `org_instance` is not an `Operation` or `Tensor`.

