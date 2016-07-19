Saves checkpoints every N steps.
- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.__init__(every_n_steps, saver, checkpoint_dir, checkpoint_basename='model.ckpt', first_n_steps=-1)` {#CheckpointSaver.__init__}

Initialize CheckpointSaver monitor.

##### Args:


*  <b>`every_n_steps`</b>: `int`, save every N steps.
*  <b>`saver`</b>: `Saver` object, used for saving.
*  <b>`checkpoint_dir`</b>: `str`, base directory for the checkpoint files.
*  <b>`checkpoint_basename`</b>: `str`, base name for the checkpoint files.
*  <b>`first_n_steps`</b>: `int`, if positive, save every step during the
    first `first_n_steps` steps.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.begin(max_steps=None, init_step=None)` {#CheckpointSaver.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.
*  <b>`init_step`</b>: `int`, step at which this training will start.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


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

#### `tf.contrib.learn.monitors.CheckpointSaver.every_n_post_step(step, session)` {#CheckpointSaver.every_n_post_step}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.every_n_step_begin(step)` {#CheckpointSaver.every_n_step_begin}

Callback before every n'th step begins.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list` of tensors that will be evaluated at this step.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.every_n_step_end(step, outputs)` {#CheckpointSaver.every_n_step_end}

Callback after every n'th step finished.

This callback provides access to the tensors/ops evaluated at this step,
including the additional tensors for which evaluation was requested in
`step_begin`.

In addition, the callback has the opportunity to stop training by returning
`True`. This is useful for early stopping, for example.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`outputs`</b>: `dict` mapping `string` values representing tensor names to
    the value resulted from running these tensors. Values may be either
    scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

##### Returns:

  `bool`. True if training should stop.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.post_step(step, session)` {#CheckpointSaver.post_step}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.run_on_all_workers` {#CheckpointSaver.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.set_estimator(estimator)` {#CheckpointSaver.set_estimator}

A setter called automatically by the target estimator.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.step_begin(step)` {#CheckpointSaver.step_begin}

Overrides `BaseMonitor.step_begin`.

When overriding this method, you must call the super implementation.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list`, the result of every_n_step_begin, if that was called this step,
  or an empty list otherwise.

##### Raises:


*  <b>`ValueError`</b>: if called more than once during a step.


- - -

#### `tf.contrib.learn.monitors.CheckpointSaver.step_end(step, output)` {#CheckpointSaver.step_end}

Overrides `BaseMonitor.step_end`.

When overriding this method, you must call the super implementation.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`output`</b>: `dict` mapping `string` values representing tensor names to
    the value resulted from running these tensors. Values may be either
    scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

##### Returns:

  `bool`, the result of every_n_step_end, if that was called this step,
  or `False` otherwise.


