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

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.StopAtStep.step_begin(step)` {#StopAtStep.step_begin}




- - -

#### `tf.contrib.learn.monitors.StopAtStep.step_end(step, output)` {#StopAtStep.step_end}




