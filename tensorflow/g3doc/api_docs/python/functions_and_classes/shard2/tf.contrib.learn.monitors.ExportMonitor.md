Monitor that exports Estimator every N steps.
- - -

#### `tf.contrib.learn.monitors.ExportMonitor.__init__(every_n_steps, export_dir, exports_to_keep=5, signature_fn=None, default_batch_size=1)` {#ExportMonitor.__init__}

Initializes ExportMonitor.

##### Args:


*  <b>`every_n_steps`</b>: Run monitor every N steps.
*  <b>`export_dir`</b>: str, folder to export.
*  <b>`exports_to_keep`</b>: int, number of exports to keep.
*  <b>`signature_fn`</b>: Function that given `Tensor` of `Example` strings,
    `dict` of `Tensor`s for features and `dict` of `Tensor`s for predictions
    and returns default and named exporting signautres.
*  <b>`default_batch_size`</b>: Default batch size of the `Example` placeholder.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.begin(max_steps=None)` {#ExportMonitor.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.end(session=None)` {#ExportMonitor.end}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.epoch_begin(epoch)` {#ExportMonitor.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.epoch_end(epoch)` {#ExportMonitor.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.every_n_post_step(step, session)` {#ExportMonitor.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.every_n_step_begin(step)` {#ExportMonitor.every_n_step_begin}

Callback before every n'th step begins.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list` of tensors that will be evaluated at this step.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.every_n_step_end(step, outputs)` {#ExportMonitor.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.post_step(step, session)` {#ExportMonitor.post_step}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.run_on_all_workers` {#ExportMonitor.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.set_estimator(estimator)` {#ExportMonitor.set_estimator}

A setter called automatically by the target estimator.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.step_begin(step)` {#ExportMonitor.step_begin}

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

#### `tf.contrib.learn.monitors.ExportMonitor.step_end(step, output)` {#ExportMonitor.step_end}

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


