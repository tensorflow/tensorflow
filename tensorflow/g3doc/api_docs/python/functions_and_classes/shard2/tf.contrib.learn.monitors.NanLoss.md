NaN Loss monitor.

Monitors loss and stops training if loss is NaN.
Can either fail with exception or just stop training.
- - -

#### `tf.contrib.learn.monitors.NanLoss.__init__(loss_tensor, every_n_steps=100, fail_on_nan_loss=True)` {#NanLoss.__init__}

Initializes NanLoss monitor.

##### Args:


*  <b>`loss_tensor`</b>: `Tensor`, the loss tensor.
*  <b>`every_n_steps`</b>: `int`, run check every this many steps.
*  <b>`fail_on_nan_loss`</b>: `bool`, whether to raise exception when loss is NaN.


- - -

#### `tf.contrib.learn.monitors.NanLoss.begin(max_steps=None)` {#NanLoss.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.NanLoss.end(session=None)` {#NanLoss.end}




- - -

#### `tf.contrib.learn.monitors.NanLoss.epoch_begin(epoch)` {#NanLoss.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.NanLoss.epoch_end(epoch)` {#NanLoss.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.NanLoss.every_n_post_step(step, session)` {#NanLoss.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.NanLoss.every_n_step_begin(step)` {#NanLoss.every_n_step_begin}




- - -

#### `tf.contrib.learn.monitors.NanLoss.every_n_step_end(step, outputs)` {#NanLoss.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.NanLoss.post_step(step, session)` {#NanLoss.post_step}




- - -

#### `tf.contrib.learn.monitors.NanLoss.run_on_all_workers` {#NanLoss.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.NanLoss.set_estimator(estimator)` {#NanLoss.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.NanLoss.step_begin(step)` {#NanLoss.step_begin}

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

#### `tf.contrib.learn.monitors.NanLoss.step_end(step, output)` {#NanLoss.step_end}

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


