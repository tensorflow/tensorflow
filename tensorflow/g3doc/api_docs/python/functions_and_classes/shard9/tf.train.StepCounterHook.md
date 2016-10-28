Steps per second monitor.
- - -

#### `tf.train.StepCounterHook.__init__(every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None)` {#StepCounterHook.__init__}




- - -

#### `tf.train.StepCounterHook.after_run(run_context, run_values)` {#StepCounterHook.after_run}




- - -

#### `tf.train.StepCounterHook.before_run(run_context)` {#StepCounterHook.before_run}




- - -

#### `tf.train.StepCounterHook.begin()` {#StepCounterHook.begin}




- - -

#### `tf.train.StepCounterHook.end(session)` {#StepCounterHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


