#### `tf.train.Server.create_local_server(start=True)` {#Server.create_local_server}

Creates a new single-process cluster running on the local host.

This method is a convenience wrapper for creating a
`tf.train.Server` with a `tf.train.ServerDef` that specifies a
single-process cluster containing a single task in a job called
`"local"`.

##### Args:


*  <b>`start`</b>: (Optional.) Boolean, indicating whether to start the server after
    creating it. Defaults to `True`.

##### Returns:

  A local `tf.train.Server`.

