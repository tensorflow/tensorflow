An in-process TensorFlow server, for use in distributed training.

A `tf.train.Server` instance encapsulates a set of devices and a
[`tf.Session`](../../api_docs/python/client.md#Session) target that
can participate in distributed training. A server belongs to a
cluster (specified by a [`tf.train.ClusterSpec`](#ClusterSpec)), and
corresponds to a particular task in a named job. The server can
communicate with any other server in the same cluster.

- - -

#### `tf.train.Server.__init__(server_or_cluster_def, job_name=None, task_index=None, protocol=None, start=True)` {#Server.__init__}

Creates a new server with the given definition.

The `job_name`, `task_index`, and `protocol` arguments are optional, and
override any information provided in `server_or_cluster_def`.

##### Args:


*  <b>`server_or_cluster_def`</b>: A `tf.train.ServerDef` or
    `tf.train.ClusterDef` protocol buffer, or a
    `tf.train.ClusterSpec` object, describing the server to be
    created and/or the cluster of which it is a member.
*  <b>`job_name`</b>: (Optional.) Specifies the name of the job of which the server
    is a member. Defaults to the value in `server_or_cluster_def`, if
    specified.
*  <b>`task_index`</b>: (Optional.) Specifies the task index of the server in its
    job. Defaults to the value in `server_or_cluster_def`, if specified.
    Otherwise defaults to 0 if the server's job has only one task.
*  <b>`protocol`</b>: (Optional.) Specifies the protocol to be used by the server.
    Acceptable values include `"grpc"`. Defaults to the value in
    `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.
*  <b>`start`</b>: (Optional.) Boolean, indicating whether to start the server
    after creating it. Defaults to `True`.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    creating the TensorFlow server.


- - -

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


- - -

#### `tf.train.Server.target` {#Server.target}

Returns the target for a `tf.Session` to connect to this server.

To create a
[`tf.Session`](../../api_docs/python/client.md#Session) that
connects to this server, use the following snippet:

```python
server = tf.train.Server(...)
with tf.Session(server.target):
  # ...
```

##### Returns:

  A string containing a session target for this server.



- - -

#### `tf.train.Server.start()` {#Server.start}

Starts this server.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    starting the TensorFlow server.


- - -

#### `tf.train.Server.join()` {#Server.join}

Blocks until the server has shut down.

This method currently blocks forever.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    joining the TensorFlow server.


