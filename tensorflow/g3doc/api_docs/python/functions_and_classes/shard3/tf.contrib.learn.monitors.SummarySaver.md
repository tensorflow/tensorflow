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


