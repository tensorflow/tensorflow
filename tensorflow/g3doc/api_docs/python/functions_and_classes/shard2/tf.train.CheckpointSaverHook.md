Saves checkpoints every N steps or seconds.
- - -

#### `tf.train.CheckpointSaverHook.__init__(checkpoint_dir, save_secs=None, save_steps=None, saver=None, checkpoint_basename='model.ckpt', scaffold=None)` {#CheckpointSaverHook.__init__}

Initialize CheckpointSaverHook monitor.

##### Args:


*  <b>`checkpoint_dir`</b>: `str`, base directory for the checkpoint files.
*  <b>`save_secs`</b>: `int`, save every N secs.
*  <b>`save_steps`</b>: `int`, save every N steps.
*  <b>`saver`</b>: `Saver` object, used for saving.
*  <b>`checkpoint_basename`</b>: `str`, base name for the checkpoint files.
*  <b>`scaffold`</b>: `Scaffold`, use to get saver object.

##### Raises:


*  <b>`ValueError`</b>: One of `save_steps` or `save_secs` should be set.
*  <b>`ValueError`</b>: Exactly one of saver or scaffold should be set.


- - -

#### `tf.train.CheckpointSaverHook.after_run(run_context, run_values)` {#CheckpointSaverHook.after_run}




- - -

#### `tf.train.CheckpointSaverHook.before_run(run_context)` {#CheckpointSaverHook.before_run}




- - -

#### `tf.train.CheckpointSaverHook.begin()` {#CheckpointSaverHook.begin}




- - -

#### `tf.train.CheckpointSaverHook.end(session)` {#CheckpointSaverHook.end}




