### `tf.train.MonitoredTrainingSession(master='', is_chief=True, checkpoint_dir=None, scaffold=None, hooks=None, chief_only_hooks=None, save_checkpoint_secs=600, save_summaries_steps=100, save_summaries_secs=None, config=None, stop_grace_period_secs=120)` {#MonitoredTrainingSession}

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
*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified, a default one is created. It's used to finalize the graph.
*  <b>`hooks`</b>: Optional list of `SessionRunHook` objects.
*  <b>`chief_only_hooks`</b>: list of `SessionRunHook` objects. Activate these hooks if
    `is_chief==True`, ignore otherwise.
*  <b>`save_checkpoint_secs`</b>: The frequency, in seconds, that a checkpoint is saved
    using a default checkpoint saver. If `save_checkpoint_secs` is set to
    `None`, then the default checkpoint saver isn't used.
*  <b>`save_summaries_steps`</b>: The frequency, in number of global steps, that the
    summaries are written to disk using a default summary saver. If both
    `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
    the default summary saver isn't used.
*  <b>`save_summaries_secs`</b>: The frequency, in secs, that the summaries are written
    to disk using a default summary saver.  If both `save_summaries_steps` and
    `save_summaries_secs` are set to `None`, then the default summary saver
    isn't used.
*  <b>`config`</b>: an instance of `tf.ConfigProto` proto used to configure the session.
    It's the `config` argument of constructor of `tf.Session`.
*  <b>`stop_grace_period_secs`</b>: Number of seconds given to threads to stop after
    `close()` has been called.

##### Returns:

  A `MonitoredSession` object.

