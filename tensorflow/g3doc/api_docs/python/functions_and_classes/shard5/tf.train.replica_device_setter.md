### `tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', merge_devices=True, cluster=None, ps_ops=None, ps_strategy=None)` {#replica_device_setter}

Return a `device function` to use when building a Graph for replicas.

Device Functions are used in `with tf.device(device_function):` statement to
automatically assign devices to `Operation` objects as they are constructed,
Device constraints are added from the inner-most context first, working
outwards. The merging behavior adds constraints to fields that are yet unset
by a more inner context. Currently the fields are (job, task, cpu/gpu).

If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.
Otherwise, the value of `ps_tasks` is derived from `cluster`.

By default, only Variable ops are placed on ps tasks, and the placement
strategy is round-robin over all ps tasks. A custom `ps_strategy` may be used
to do more intelligent placement, such as
`tf.contrib.training.GreedyLoadBalancingStrategy`.

For example,

```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```

##### Args:


*  <b>`ps_tasks`</b>: Number of tasks in the `ps` job.  Ignored if `cluster` is
    provided.
*  <b>`ps_device`</b>: String.  Device of the `ps` job.  If empty no `ps` job is used.
    Defaults to `ps`.
*  <b>`worker_device`</b>: String.  Device of the `worker` job.  If empty no `worker`
    job is used.
*  <b>`merge_devices`</b>: `Boolean`. If `True`, merges or only sets a device if the
    device constraint is completely unset. merges device specification rather
    than overriding them.
*  <b>`cluster`</b>: `ClusterDef` proto or `ClusterSpec`.
*  <b>`ps_ops`</b>: List of strings representing `Operation` types that need to be
    placed on `ps` devices.  If `None`, defaults to `["Variable"]`.
*  <b>`ps_strategy`</b>: A callable invoked for every ps `Operation` (i.e. matched by
    `ps_ops`), that takes the `Operation` and returns the ps task index to
    use.  If `None`, defaults to a round-robin strategy across all `ps`
    devices.

##### Returns:

  A function to pass to `tf.device()`.

##### Raises:

  TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer,
  or if `ps_strategy` is provided but not a callable.

