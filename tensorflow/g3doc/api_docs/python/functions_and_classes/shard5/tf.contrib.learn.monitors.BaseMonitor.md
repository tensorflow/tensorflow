Base class for Monitors.

Defines basic interfaces of Monitors.
Monitors can either be run on all workers or, more commonly, restricted
to run exclusively on the elected chief worker.
- - -

#### `tf.contrib.learn.monitors.BaseMonitor.__init__()` {#BaseMonitor.__init__}




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


