#### `tf.Session.reset(target, containers=None, config=None)` {#Session.reset}

Resets resource containers on `target`, and close all connected sessions.

A resource container is distributed across all workers in the
same cluster as `target`.  When a resource container on `target`
is reset, resources associated with that container will be cleared.
In particular, all Variables in the container will become undefined:
they lose their values and shapes.

NOTE:
(i) reset() is currently only implemented for distributed sessions.
(ii) Any sessions on the master named by `target` will be closed.

If no resource containers are provided, all containers are reset.

##### Args:


*  <b>`target`</b>: The execution engine to connect to.
*  <b>`containers`</b>: A list of resource container name strings, or `None` if all of
    all the containers are to be reset.
*  <b>`config`</b>: (Optional.) Protocol buffer with configuration options.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    resetting containers.

