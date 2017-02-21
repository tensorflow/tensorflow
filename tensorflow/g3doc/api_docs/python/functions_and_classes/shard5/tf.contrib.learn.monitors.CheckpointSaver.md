Saves checkpoints every N steps or N seconds.
- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.__init__(checkpoint_dir, save_secs=None, save_steps=None, saver=None, checkpoint_basename='model.ckpt', scaffold=None)` {#CheckpointSaver.__init__}

Initialize CheckpointSaver monitor.

##### Args:


*  <b>`checkpoint_dir`</b>: `str`, base directory for the checkpoint files.
*  <b>`save_secs`</b>: `int`, save every N secs.
*  <b>`save_steps`</b>: `int`, save every N steps.
*  <b>`saver`</b>: `Saver` object, used for saving.
*  <b>`checkpoint_basename`</b>: `str`, base name for the checkpoint files.
*  <b>`scaffold`</b>: `Scaffold`, use to get saver object.

##### Raises:


*  <b>`ValueError`</b>: If both `save_steps` and `save_secs` are not `None`.
*  <b>`ValueError`</b>: If both `save_steps` and `save_secs` are `None`.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.begin(max_steps=None)` {#CheckpointSaver.begin}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.end(session=None)` {#CheckpointSaver.end}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.epoch_begin(epoch)` {#CheckpointSaver.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.epoch_end(epoch)` {#CheckpointSaver.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.post_step(step, session)` {#CheckpointSaver.post_step}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.run_on_all_workers` {#CheckpointSaver.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.set_estimator(estimator)` {#CheckpointSaver.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.step_begin(step)` {#CheckpointSaver.step_begin}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.step_end(step, output)` {#CheckpointSaver.step_end}

Callback after training step finished.

This callback provides access to the tensors/ops evaluated at this step,
including the additional tensors for which evaluation was requested in
`step_begin`.

In addition, the callback has the opportunity to stop training by returning
`True`. This is useful for early stopping, for example.

Note that this method is not called if the call to `Session.run()` that
followed the last call to `step_begin()` failed.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`output`</b>: `dict` mapping `string` values representing tensor names to
    the value resulted from running these tensors. Values may be either
    scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

##### Returns:

  `bool`. True if training should stop.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun a step, or `step` number does not match.


