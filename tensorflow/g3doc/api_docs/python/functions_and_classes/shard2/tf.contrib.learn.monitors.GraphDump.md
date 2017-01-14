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

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.GraphDump.step_begin(step)` {#GraphDump.step_begin}




- - -

#### `tf.contrib.learn.monitors.GraphDump.step_end(step, output)` {#GraphDump.step_end}




