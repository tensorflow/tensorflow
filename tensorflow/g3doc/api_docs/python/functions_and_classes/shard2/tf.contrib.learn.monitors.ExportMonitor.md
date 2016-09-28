Monitor that exports Estimator every N steps.
- - -

#### `tf.contrib.learn.monitors.ExportMonitor.__init__(*args, **kwargs)` {#ExportMonitor.__init__}

Initializes ExportMonitor. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn and input_feature_key will both become required args.

    Args:
      every_n_steps: Run monitor every N steps.
      export_dir: str, folder to export.
      input_fn: A function that takes no argument and returns a tuple of
        (features, targets), where features is a dict of string key to `Tensor`
        and targets is a `Tensor` that's currently not used (and so can be
        `None`).
      input_feature_key: String key into the features dict returned by
        `input_fn` that corresponds to the raw `Example` strings `Tensor` that
        the exported model will take as input.
      exports_to_keep: int, number of exports to keep.
      signature_fn: Function that returns a default signature and a named
        signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
        for features and `dict` of `Tensor`s for predictions.
      default_batch_size: Default batch size of the `Example` placeholder.

    Raises:
      ValueError: If `input_fn` and `input_feature_key` are not both defined or
        are not both `None`.


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

#### `tf.contrib.learn.monitors.ExportMonitor.export_dir` {#ExportMonitor.export_dir}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.exports_to_keep` {#ExportMonitor.exports_to_keep}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.last_export_dir` {#ExportMonitor.last_export_dir}

Returns the directory containing the last completed export.

##### Returns:

  The string path to the exported directory. NB: this functionality was
  added on 2016/09/25; clients that depend on the return value may need
  to handle the case where this function returns None because the
  estimator being fitted does not yet return a value during export.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.post_step(step, session)` {#ExportMonitor.post_step}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.run_on_all_workers` {#ExportMonitor.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.ExportMonitor.set_estimator(estimator)` {#ExportMonitor.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.ExportMonitor.signature_fn` {#ExportMonitor.signature_fn}




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


