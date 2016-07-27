Prints given tensors every N steps.

This is an `EveryN` monitor and has consistent semantic for `every_n`
and `first_n`.

The tensors will be printed to the log, with `INFO` severity.
- - -

#### `tf.contrib.learn.monitors.PrintTensor.__init__(tensor_names, every_n=100, first_n=1)` {#PrintTensor.__init__}

Initializes a PrintTensor monitor.

##### Args:


*  <b>`tensor_names`</b>: `dict` of tag to tensor names or
      `iterable` of tensor names (strings).
*  <b>`every_n`</b>: `int`, print every N steps. See `PrintN.`
*  <b>`first_n`</b>: `int`, also print the first N steps. See `PrintN.`


- - -

#### `tf.contrib.learn.monitors.PrintTensor.begin(max_steps=None)` {#PrintTensor.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.PrintTensor.end(session=None)` {#PrintTensor.end}




- - -

#### `tf.contrib.learn.monitors.PrintTensor.epoch_begin(epoch)` {#PrintTensor.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.PrintTensor.epoch_end(epoch)` {#PrintTensor.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.PrintTensor.every_n_post_step(step, session)` {#PrintTensor.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.PrintTensor.every_n_step_begin(step)` {#PrintTensor.every_n_step_begin}




- - -

#### `tf.contrib.learn.monitors.PrintTensor.every_n_step_end(step, outputs)` {#PrintTensor.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.PrintTensor.post_step(step, session)` {#PrintTensor.post_step}




- - -

#### `tf.contrib.learn.monitors.PrintTensor.run_on_all_workers` {#PrintTensor.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.PrintTensor.set_estimator(estimator)` {#PrintTensor.set_estimator}

A setter called automatically by the target estimator.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.PrintTensor.step_begin(step)` {#PrintTensor.step_begin}

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

#### `tf.contrib.learn.monitors.PrintTensor.step_end(step, output)` {#PrintTensor.step_end}

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


