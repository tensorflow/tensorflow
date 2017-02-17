### `tf.contrib.copy_graph.get_copied_op(org_instance, graph, scope='')` {#get_copied_op}

Given an `Operation` instance from some `Graph`, returns
its namesake from `graph`, under the specified scope
(default `""`).

If a copy of `org_instance` is present in `graph` under the given
`scope`, it will be returned.

Args:
org_instance: An `Operation` from some `Graph`.
graph: The `Graph` to be searched for a copr of `org_instance`.
scope: The scope `org_instance` is present in.

##### Returns:

    The `Operation` copy from `graph`.

