Captures a variable's values into a collection.

This monitor is useful for unit testing. You should exercise caution when
using this monitor in production, since it never discards values.

This is an `EveryN` monitor and has consistent semantic for `every_n`
and `first_n`.
- - -

#### `tf.contrib.learn.monitors.CaptureVariable.__init__(var_name, every_n=100, first_n=1)` {#CaptureVariable.__init__}

Initializes a CaptureVariable monitor.

##### Args:


*  <b>`var_name`</b>: `string`. The variable name, including suffix (typically ":0").
*  <b>`every_n`</b>: `int`, print every N steps. See `PrintN.`
*  <b>`first_n`</b>: `int`, also print the first N steps. See `PrintN.`


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.begin(max_steps=None)` {#CaptureVariable.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.end(session=None)` {#CaptureVariable.end}




- - -

#### `tf.contrib.learn.monitors.CaptureVariable.epoch_begin(epoch)` {#CaptureVariable.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.epoch_end(epoch)` {#CaptureVariable.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.every_n_post_step(step, session)` {#CaptureVariable.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.every_n_step_begin(step)` {#CaptureVariable.every_n_step_begin}




- - -

#### `tf.contrib.learn.monitors.CaptureVariable.every_n_step_end(step, outputs)` {#CaptureVariable.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.CaptureVariable.post_step(step, session)` {#CaptureVariable.post_step}




- - -

#### `tf.contrib.learn.monitors.CaptureVariable.run_on_all_workers` {#CaptureVariable.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.CaptureVariable.set_estimator(estimator)` {#CaptureVariable.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.step_begin(step)` {#CaptureVariable.step_begin}

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

#### `tf.contrib.learn.monitors.CaptureVariable.step_end(step, output)` {#CaptureVariable.step_end}

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


- - -

#### `tf.contrib.learn.monitors.CaptureVariable.values` {#CaptureVariable.values}

Returns the values captured so far.

##### Returns:

  `dict` mapping `int` step numbers to that values of the variable at the
      respective step.


