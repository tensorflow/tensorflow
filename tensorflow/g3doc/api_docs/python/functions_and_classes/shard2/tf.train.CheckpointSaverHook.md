Saves checkpoints every N steps or seconds.
- - -

#### `tf.train.CheckpointSaverHook.__init__(checkpoint_dir, save_secs=None, save_steps=None, saver=None, checkpoint_basename='model.ckpt', scaffold=None, listeners=None)` {#CheckpointSaverHook.__init__}

Initialize CheckpointSaverHook monitor.

##### Args:


*  <b>`checkpoint_dir`</b>: `str`, base directory for the checkpoint files.
*  <b>`save_secs`</b>: `int`, save every N secs.
*  <b>`save_steps`</b>: `int`, save every N steps.
*  <b>`saver`</b>: `Saver` object, used for saving.
*  <b>`checkpoint_basename`</b>: `str`, base name for the checkpoint files.
*  <b>`scaffold`</b>: `Scaffold`, use to get saver object.
*  <b>`listeners`</b>: List of `CheckpointSaverListener` subclass instances.
    Used for callbacks that run immediately after the corresponding
    CheckpointSaverHook callbacks, only in steps where the
    CheckpointSaverHook was triggered.

##### Raises:


*  <b>`ValueError`</b>: One of `save_steps` or `save_secs` should be set.
*  <b>`ValueError`</b>: Exactly one of saver or scaffold should be set.


- - -

#### `tf.train.CheckpointSaverHook.after_create_session(session, coord)` {#CheckpointSaverHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.
*  <b>`coord`</b>: A Coordinator object which keeps track of all threads.


- - -

#### `tf.train.CheckpointSaverHook.after_run(run_context, run_values)` {#CheckpointSaverHook.after_run}




- - -

#### `tf.train.CheckpointSaverHook.before_run(run_context)` {#CheckpointSaverHook.before_run}




- - -

#### `tf.train.CheckpointSaverHook.begin()` {#CheckpointSaverHook.begin}




- - -

#### `tf.train.CheckpointSaverHook.end(session)` {#CheckpointSaverHook.end}




