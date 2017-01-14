Steps per second monitor.
- - -

#### `tf.contrib.learn.monitors.StepCounter.__init__(every_n_steps=100, output_dir=None, summary_writer=None)` {#StepCounter.__init__}




- - -

#### `tf.contrib.learn.monitors.StepCounter.begin(max_steps=None)` {#StepCounter.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.StepCounter.end(session=None)` {#StepCounter.end}




- - -

#### `tf.contrib.learn.monitors.StepCounter.epoch_begin(epoch)` {#StepCounter.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.StepCounter.epoch_end(epoch)` {#StepCounter.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.StepCounter.every_n_post_step(step, session)` {#StepCounter.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.StepCounter.every_n_step_begin(step)` {#StepCounter.every_n_step_begin}

Callback before every n'th step begins.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list` of tensors that will be evaluated at this step.


- - -

#### `tf.contrib.learn.monitors.StepCounter.every_n_step_end(current_step, outputs)` {#StepCounter.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.StepCounter.post_step(step, session)` {#StepCounter.post_step}




- - -

#### `tf.contrib.learn.monitors.StepCounter.run_on_all_workers` {#StepCounter.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.StepCounter.set_estimator(estimator)` {#StepCounter.set_estimator}




- - -

#### `tf.contrib.learn.monitors.StepCounter.step_begin(step)` {#StepCounter.step_begin}

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

#### `tf.contrib.learn.monitors.StepCounter.step_end(step, output)` {#StepCounter.step_end}

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


