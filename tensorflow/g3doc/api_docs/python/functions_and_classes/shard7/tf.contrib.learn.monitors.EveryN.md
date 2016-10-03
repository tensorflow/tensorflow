Base class for monitors that execute callbacks every N steps.

This class adds three new callbacks:
  - every_n_step_begin
  - every_n_step_end
  - every_n_post_step

The callbacks are executed every n steps, or optionally every step for the
first m steps, where m and n can both be user-specified.

When extending this class, note that if you wish to use any of the
`BaseMonitor` callbacks, you must call their respective super implementation:

  def step_begin(self, step):
    super(ExampleMonitor, self).step_begin(step)
    return []

Failing to call the super implementation will cause unpredictable behavior.

The `every_n_post_step()` callback is also called after the last step if it
was not already called through the regular conditions.  Note that
`every_n_step_begin()` and `every_n_step_end()` do not receive that special
treatment.
- - -

#### `tf.contrib.learn.monitors.EveryN.__init__(every_n_steps=100, first_n_steps=1)` {#EveryN.__init__}

Initializes an `EveryN` monitor.

##### Args:


*  <b>`every_n_steps`</b>: `int`, the number of steps to allow between callbacks.
*  <b>`first_n_steps`</b>: `int`, specifying the number of initial steps during
    which the callbacks will always be executed, regardless of the value
    of `every_n_steps`. Note that this value is relative to the global step


- - -

#### `tf.contrib.learn.monitors.EveryN.begin(max_steps=None)` {#EveryN.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.EveryN.end(session=None)` {#EveryN.end}




- - -

#### `tf.contrib.learn.monitors.EveryN.epoch_begin(epoch)` {#EveryN.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.EveryN.epoch_end(epoch)` {#EveryN.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.EveryN.every_n_post_step(step, session)` {#EveryN.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.EveryN.every_n_step_begin(step)` {#EveryN.every_n_step_begin}

Callback before every n'th step begins.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list` of tensors that will be evaluated at this step.


- - -

#### `tf.contrib.learn.monitors.EveryN.every_n_step_end(step, outputs)` {#EveryN.every_n_step_end}

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

#### `tf.contrib.learn.monitors.EveryN.post_step(step, session)` {#EveryN.post_step}




- - -

#### `tf.contrib.learn.monitors.EveryN.run_on_all_workers` {#EveryN.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.EveryN.set_estimator(estimator)` {#EveryN.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.EveryN.step_begin(step)` {#EveryN.step_begin}

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

#### `tf.contrib.learn.monitors.EveryN.step_end(step, output)` {#EveryN.step_end}

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


