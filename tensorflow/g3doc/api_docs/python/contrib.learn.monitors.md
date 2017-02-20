<!-- This file is machine generated: DO NOT EDIT! -->

# Monitors (contrib)
[TOC]

Monitors instrument the training process.

See the @{$python/contrib.learn.monitors} guide.

- - -

### `tf.contrib.learn.monitors.get_default_monitors(loss_op=None, summary_op=None, save_summary_steps=100, output_dir=None, summary_writer=None)` {#get_default_monitors}

Returns a default set of typically-used monitors.

##### Args:


*  <b>`loss_op`</b>: `Tensor`, the loss tensor. This will be printed using `PrintTensor`
      at the default interval.
*  <b>`summary_op`</b>: See `SummarySaver`.
*  <b>`save_summary_steps`</b>: See `SummarySaver`.
*  <b>`output_dir`</b>: See `SummarySaver`.
*  <b>`summary_writer`</b>: See `SummarySaver`.

##### Returns:

  `list` of monitors.


- - -

### `class tf.contrib.learn.monitors.BaseMonitor` {#BaseMonitor}

Base class for Monitors.

Defines basic interfaces of Monitors.
Monitors can either be run on all workers or, more commonly, restricted
to run exclusively on the elected chief worker.
- - -

#### `tf.contrib.learn.monitors.BaseMonitor.__init__(*args, **kwargs)` {#BaseMonitor.__init__}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-05.
Instructions for updating:
Monitors are deprecated. Please use tf.train.SessionRunHook.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.begin(max_steps=None)` {#BaseMonitor.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.end(session=None)` {#BaseMonitor.end}

Callback at the end of training/evaluation.

##### Args:


*  <b>`session`</b>: A `tf.Session` object that can be used to run ops.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun a run.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.epoch_begin(epoch)` {#BaseMonitor.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.epoch_end(epoch)` {#BaseMonitor.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.post_step(step, session)` {#BaseMonitor.post_step}

Callback after the step is finished.

Called after step_end and receives session to perform extra session.run
calls. If failure occurred in the process, will be called as well.

##### Args:


*  <b>`step`</b>: `int`, global step of the model.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.run_on_all_workers` {#BaseMonitor.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.BaseMonitor.set_estimator(estimator)` {#BaseMonitor.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.step_begin(step)` {#BaseMonitor.step_begin}

Callback before training step begins.

You may use this callback to request evaluation of additional tensors
in the graph.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  List of `Tensor` objects or string tensor names to be run.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a step, or `step` < 0, or
      `step` > `max_steps`.


- - -

#### `tf.contrib.learn.monitors.BaseMonitor.step_end(step, output)` {#BaseMonitor.step_end}

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



- - -

### `class tf.contrib.learn.monitors.CaptureVariable` {#CaptureVariable}

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



- - -

### `class tf.contrib.learn.monitors.CheckpointSaver` {#CheckpointSaver}

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



- - -

### `class tf.contrib.learn.monitors.EveryN` {#EveryN}

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



- - -

### `class tf.contrib.learn.monitors.ExportMonitor` {#ExportMonitor}

Monitor that exports Estimator every N steps.
- - -

#### `tf.contrib.learn.monitors.ExportMonitor.__init__(*args, **kwargs)` {#ExportMonitor.__init__}

Initializes ExportMonitor. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-23.
Instructions for updating:
The signature of the input_fn accepted by export is changing to be consistent with what's used by tf.Learn Estimator's train/evaluate. input_fn (and in most cases, input_feature_key) will both become required args.

##### Args:


*  <b>`every_n_steps`</b>: Run monitor every N steps.
*  <b>`export_dir`</b>: str, folder to export.
*  <b>`input_fn`</b>: A function that takes no argument and returns a tuple of
    (features, labels), where features is a dict of string key to `Tensor`
    and labels is a `Tensor` that's currently not used (and so can be
    `None`).
*  <b>`input_feature_key`</b>: String key into the features dict returned by
    `input_fn` that corresponds to the raw `Example` strings `Tensor` that
    the exported model will take as input. Should be `None` if and only if
    you're passing in a `signature_fn` that does not use the first arg
    (`Tensor` of `Example` strings).
*  <b>`exports_to_keep`</b>: int, number of exports to keep.
*  <b>`signature_fn`</b>: Function that returns a default signature and a named
    signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
    for features and `dict` of `Tensor`s for predictions.
*  <b>`default_batch_size`</b>: Default batch size of the `Example` placeholder.

##### Raises:


*  <b>`ValueError`</b>: If `input_fn` and `input_feature_key` are not both defined or
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



- - -

### `class tf.contrib.learn.monitors.GraphDump` {#GraphDump}

Dumps almost all tensors in the graph at every step.

Note, this is very expensive, prefer `PrintTensor` in production.
- - -

#### `tf.contrib.learn.monitors.GraphDump.__init__(ignore_ops=None)` {#GraphDump.__init__}

Initializes GraphDump monitor.

##### Args:


*  <b>`ignore_ops`</b>: `list` of `string`. Names of ops to ignore.
      If None, `GraphDump.IGNORE_OPS` is used.


- - -

#### `tf.contrib.learn.monitors.GraphDump.begin(max_steps=None)` {#GraphDump.begin}




- - -

#### `tf.contrib.learn.monitors.GraphDump.compare(other_dump, step, atol=1e-06)` {#GraphDump.compare}

Compares two `GraphDump` monitors and returns differences.

##### Args:


*  <b>`other_dump`</b>: Another `GraphDump` monitor.
*  <b>`step`</b>: `int`, step to compare on.
*  <b>`atol`</b>: `float`, absolute tolerance in comparison of floating arrays.

##### Returns:

  Returns tuple:

*  <b>`matched`</b>: `list` of keys that matched.
*  <b>`non_matched`</b>: `dict` of keys to tuple of 2 mismatched values.

##### Raises:


*  <b>`ValueError`</b>: if a key in `data` is missing from `other_dump` at `step`.


- - -

#### `tf.contrib.learn.monitors.GraphDump.data` {#GraphDump.data}




- - -

#### `tf.contrib.learn.monitors.GraphDump.end(session=None)` {#GraphDump.end}

Callback at the end of training/evaluation.

##### Args:


*  <b>`session`</b>: A `tf.Session` object that can be used to run ops.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun a run.


- - -

#### `tf.contrib.learn.monitors.GraphDump.epoch_begin(epoch)` {#GraphDump.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.GraphDump.epoch_end(epoch)` {#GraphDump.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.GraphDump.post_step(step, session)` {#GraphDump.post_step}

Callback after the step is finished.

Called after step_end and receives session to perform extra session.run
calls. If failure occurred in the process, will be called as well.

##### Args:


*  <b>`step`</b>: `int`, global step of the model.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.GraphDump.run_on_all_workers` {#GraphDump.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.GraphDump.set_estimator(estimator)` {#GraphDump.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.GraphDump.step_begin(step)` {#GraphDump.step_begin}




- - -

#### `tf.contrib.learn.monitors.GraphDump.step_end(step, output)` {#GraphDump.step_end}





- - -

### `class tf.contrib.learn.monitors.LoggingTrainable` {#LoggingTrainable}

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

If the estimator is locked, this method does nothing.

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



- - -

### `class tf.contrib.learn.monitors.NanLoss` {#NanLoss}

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



- - -

### `class tf.contrib.learn.monitors.PrintTensor` {#PrintTensor}

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

If the estimator is locked, this method does nothing.

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



- - -

### `class tf.contrib.learn.monitors.StepCounter` {#StepCounter}

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



- - -

### `class tf.contrib.learn.monitors.StopAtStep` {#StopAtStep}

Monitor to request stop at a specified step.
- - -

#### `tf.contrib.learn.monitors.StopAtStep.__init__(num_steps=None, last_step=None)` {#StopAtStep.__init__}

Create a StopAtStep monitor.

This monitor requests stop after either a number of steps have been
executed or a last step has been reached.  Only of the two options can be
specified.

if `num_steps` is specified, it indicates the number of steps to execute
after `begin()` is called.  If instead `last_step` is specified, it
indicates the last step we want to execute, as passed to the `step_begin()`
call.

##### Args:


*  <b>`num_steps`</b>: Number of steps to execute.
*  <b>`last_step`</b>: Step after which to stop.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.begin(max_steps=None)` {#StopAtStep.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.end(session=None)` {#StopAtStep.end}

Callback at the end of training/evaluation.

##### Args:


*  <b>`session`</b>: A `tf.Session` object that can be used to run ops.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun a run.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.epoch_begin(epoch)` {#StopAtStep.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.epoch_end(epoch)` {#StopAtStep.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.post_step(step, session)` {#StopAtStep.post_step}

Callback after the step is finished.

Called after step_end and receives session to perform extra session.run
calls. If failure occurred in the process, will be called as well.

##### Args:


*  <b>`step`</b>: `int`, global step of the model.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.run_on_all_workers` {#StopAtStep.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.StopAtStep.set_estimator(estimator)` {#StopAtStep.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.step_begin(step)` {#StopAtStep.step_begin}




- - -

#### `tf.contrib.learn.monitors.StopAtStep.step_end(step, output)` {#StopAtStep.step_end}





- - -

### `class tf.contrib.learn.monitors.SummarySaver` {#SummarySaver}

Saves summaries every N steps.
- - -

#### `tf.contrib.learn.monitors.SummarySaver.__init__(summary_op, save_steps=100, output_dir=None, summary_writer=None, scaffold=None)` {#SummarySaver.__init__}

Initializes a `SummarySaver` monitor.

##### Args:


*  <b>`summary_op`</b>: `Tensor` of type `string`. A serialized `Summary` protocol
      buffer, as output by TF summary methods like `summary.scalar` or
      `summary.merge_all`.
*  <b>`save_steps`</b>: `int`, save summaries every N steps. See `EveryN`.
*  <b>`output_dir`</b>: `string`, the directory to save the summaries to. Only used
      if no `summary_writer` is supplied.
*  <b>`summary_writer`</b>: `SummaryWriter`. If `None` and an `output_dir` was passed,
      one will be created accordingly.
*  <b>`scaffold`</b>: `Scaffold` to get summary_op if it's not provided.


- - -

#### `tf.contrib.learn.monitors.SummarySaver.begin(max_steps=None)` {#SummarySaver.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.SummarySaver.end(session=None)` {#SummarySaver.end}




- - -

#### `tf.contrib.learn.monitors.SummarySaver.epoch_begin(epoch)` {#SummarySaver.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.SummarySaver.epoch_end(epoch)` {#SummarySaver.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.SummarySaver.every_n_post_step(step, session)` {#SummarySaver.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.SummarySaver.every_n_step_begin(step)` {#SummarySaver.every_n_step_begin}




- - -

#### `tf.contrib.learn.monitors.SummarySaver.every_n_step_end(step, outputs)` {#SummarySaver.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.SummarySaver.post_step(step, session)` {#SummarySaver.post_step}




- - -

#### `tf.contrib.learn.monitors.SummarySaver.run_on_all_workers` {#SummarySaver.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.SummarySaver.set_estimator(estimator)` {#SummarySaver.set_estimator}




- - -

#### `tf.contrib.learn.monitors.SummarySaver.step_begin(step)` {#SummarySaver.step_begin}

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

#### `tf.contrib.learn.monitors.SummarySaver.step_end(step, output)` {#SummarySaver.step_end}

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

### `class tf.contrib.learn.monitors.ValidationMonitor` {#ValidationMonitor}

Runs evaluation of a given estimator, at most every N steps.

Note that the evaluation is done based on the saved checkpoint, which will
usually be older than the current step.

Can do early stopping on validation metrics if `early_stopping_rounds` is
provided.
- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.__init__(x=None, y=None, input_fn=None, batch_size=None, eval_steps=None, every_n_steps=100, metrics=None, hooks=None, early_stopping_rounds=None, early_stopping_metric='loss', early_stopping_metric_minimize=True, name=None)` {#ValidationMonitor.__init__}

Initializes a ValidationMonitor.

##### Args:


*  <b>`x`</b>: See `BaseEstimator.evaluate`.
*  <b>`y`</b>: See `BaseEstimator.evaluate`.
*  <b>`input_fn`</b>: See `BaseEstimator.evaluate`.
*  <b>`batch_size`</b>: See `BaseEstimator.evaluate`.
*  <b>`eval_steps`</b>: See `BaseEstimator.evaluate`.
*  <b>`every_n_steps`</b>: Check for new checkpoints to evaluate every N steps. If a
      new checkpoint is found, it is evaluated. See `EveryN`.
*  <b>`metrics`</b>: See `BaseEstimator.evaluate`.
*  <b>`hooks`</b>: A list of `SessionRunHook` hooks to pass to the
    `Estimator`'s `evaluate` function.
*  <b>`early_stopping_rounds`</b>: `int`. If the metric indicated by
      `early_stopping_metric` does not change according to
      `early_stopping_metric_minimize` for this many steps, then training
      will be stopped.
*  <b>`early_stopping_metric`</b>: `string`, name of the metric to check for early
      stopping.
*  <b>`early_stopping_metric_minimize`</b>: `bool`, True if `early_stopping_metric` is
      expected to decrease (thus early stopping occurs when this metric
      stops decreasing), False if `early_stopping_metric` is expected to
      increase. Typically, `early_stopping_metric_minimize` is True for
      loss metrics like mean squared error, and False for performance
      metrics like accuracy.
*  <b>`name`</b>: See `BaseEstimator.evaluate`.

##### Raises:


*  <b>`ValueError`</b>: If both x and input_fn are provided.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.begin(max_steps=None)` {#ValidationMonitor.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.best_step` {#ValidationMonitor.best_step}

Returns the step at which the best early stopping metric was found.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.best_value` {#ValidationMonitor.best_value}

Returns the best early stopping metric value found so far.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.early_stopped` {#ValidationMonitor.early_stopped}

Returns True if this monitor caused an early stop.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.end(session=None)` {#ValidationMonitor.end}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.epoch_begin(epoch)` {#ValidationMonitor.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.epoch_end(epoch)` {#ValidationMonitor.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.every_n_post_step(step, session)` {#ValidationMonitor.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.every_n_step_begin(step)` {#ValidationMonitor.every_n_step_begin}

Callback before every n'th step begins.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list` of tensors that will be evaluated at this step.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.every_n_step_end(step, outputs)` {#ValidationMonitor.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.post_step(step, session)` {#ValidationMonitor.post_step}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.run_on_all_workers` {#ValidationMonitor.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.set_estimator(estimator)` {#ValidationMonitor.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.step_begin(step)` {#ValidationMonitor.step_begin}

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

#### `tf.contrib.learn.monitors.ValidationMonitor.step_end(step, output)` {#ValidationMonitor.step_end}

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




## Other Functions and Classes
- - -

### `class tf.contrib.learn.monitors.RunHookAdapterForMonitors` {#RunHookAdapterForMonitors}

Wraps monitors into a SessionRunHook.
- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.__init__(monitors)` {#RunHookAdapterForMonitors.__init__}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.after_create_session(session, coord)` {#RunHookAdapterForMonitors.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.
*  <b>`coord`</b>: A Coordinator object which keeps track of all threads.


- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.after_run(run_context, run_values)` {#RunHookAdapterForMonitors.after_run}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.before_run(run_context)` {#RunHookAdapterForMonitors.before_run}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.begin()` {#RunHookAdapterForMonitors.begin}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.end(session)` {#RunHookAdapterForMonitors.end}





- - -

### `class tf.contrib.learn.monitors.SummaryWriterCache` {#SummaryWriterCache}

Cache for file writers.

This class caches file writers, one per directory.
- - -

#### `tf.contrib.learn.monitors.SummaryWriterCache.clear()` {#SummaryWriterCache.clear}

Clear cached summary writers. Currently only used for unit tests.


- - -

#### `tf.contrib.learn.monitors.SummaryWriterCache.get(logdir)` {#SummaryWriterCache.get}

Returns the FileWriter for the specified directory.

##### Args:


*  <b>`logdir`</b>: str, name of the directory.

##### Returns:

  A `FileWriter`.



- - -

### `tf.contrib.learn.monitors.replace_monitors_with_hooks(monitors_or_hooks, estimator)` {#replace_monitors_with_hooks}

Wraps monitors with a hook.

`Monitor` is deprecated in favor of `SessionRunHook`. If you're using a
monitor, you can wrap it with a hook using function. It is recommended to
implement hook version of your monitor.

##### Args:


*  <b>`monitors_or_hooks`</b>: A `list` may contain both monitors and hooks.
*  <b>`estimator`</b>: An `Estimator` that monitor will be used with.

##### Returns:

  Returns a list of hooks. If there is any monitor in the given list, it is
  replaced by a hook.


