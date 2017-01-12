Steps per second monitor.
- - -

#### `tf.train.StepCounterHook.__init__(every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None)` {#StepCounterHook.__init__}




- - -

#### `tf.train.StepCounterHook.after_create_session(session)` {#StepCounterHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


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


