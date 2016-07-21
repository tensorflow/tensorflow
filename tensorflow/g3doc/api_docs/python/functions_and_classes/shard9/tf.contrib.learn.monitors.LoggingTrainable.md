Writes trainable variable values into log every N steps.

Write the tensors in trainable variables `every_n` steps,
starting with the `first_n`th step.
- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.__init__(scope=None, every_n=100, first_n=1)` {#LoggingTrainable.__init__}

Initializes LoggingTrainable monitor.

##### Args:


*  <b>`scope`</b>: An optional string to match variable names using re.match.
*  <b>`every_n`</b>: Print every N steps.
*  <b>`first_n`</b>: Print first N steps.


- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.begin(max_steps=None)` {#LoggingTrainable.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.end(session=None)` {#LoggingTrainable.end}




- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.epoch_begin(epoch)` {#LoggingTrainable.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.epoch_end(epoch)` {#LoggingTrainable.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.every_n_post_step(step, session)` {#LoggingTrainable.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.every_n_step_begin(step)` {#LoggingTrainable.every_n_step_begin}




- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.every_n_step_end(step, outputs)` {#LoggingTrainable.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.post_step(step, session)` {#LoggingTrainable.post_step}




- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.run_on_all_workers` {#LoggingTrainable.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.set_estimator(estimator)` {#LoggingTrainable.set_estimator}

A setter called automatically by the target estimator.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.LoggingTrainable.step_begin(step)` {#LoggingTrainable.step_begin}

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

#### `tf.contrib.learn.monitors.LoggingTrainable.step_end(step, output)` {#LoggingTrainable.step_end}

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


