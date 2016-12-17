Monitor to request stop at a specified step.
- - -

#### `tf.train.StopAtStepHook.__init__(num_steps=None, last_step=None)` {#StopAtStepHook.__init__}

Create a StopAtStep Hook.

This hook requests stop after either a number of steps have been
executed or a last step has been reached.  Only of the two options can be
specified.

if `num_steps` is specified, it indicates the number of steps to execute
after `begin()` is called.  If instead `last_step` is specified, it
indicates the last step we want to execute, as passed to the `after_run()`
call.

##### Args:


*  <b>`num_steps`</b>: Number of steps to execute.
*  <b>`last_step`</b>: Step after which to stop.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.train.StopAtStepHook.after_create_session(session)` {#StopAtStepHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will be called as a result of recovering a wrapped session,
    instead of at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.StopAtStepHook.after_run(run_context, run_values)` {#StopAtStepHook.after_run}




- - -

#### `tf.train.StopAtStepHook.before_run(run_context)` {#StopAtStepHook.before_run}




- - -

#### `tf.train.StopAtStepHook.begin()` {#StopAtStepHook.begin}




- - -

#### `tf.train.StopAtStepHook.end(session)` {#StopAtStepHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


