### `tf.train.MonitoredTrainingSession(master='', is_chief=True, checkpoint_dir=None, hooks=None, scaffold=None, config=None)` {#MonitoredTrainingSession}

Creates a `MonitoredSession` for training.

For a chief, this utility sets proper session initializer/restorer. It also
creates hooks related to checkpoint and summary saving. For workers, this
utility sets proper session creator which waits for the chief to
inialize/restore.


##### Args:


*  <b>`master`</b>: `String` the TensorFlow master to use.
*  <b>`is_chief`</b>: If `True`, it will take care of initialization and recovery the
    underlying TensorFlow session. If `False`, it will wait on a chief to
    initialize or recover the TensorFlow session.
*  <b>`checkpoint_dir`</b>: A string.  Optional path to a directory where to restore
    variables.
*  <b>`hooks`</b>: Optional list of `SessionRunHook` objects.
*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified, a default one is created. It's used to finalize the graph.
*  <b>`config`</b>: `ConfigProto` proto used to configure the session.

##### Returns:

  A `MonitoredSession` object.

