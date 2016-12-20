Delay execution until global step reaches to wait_until_step.

This hook delays execution until global step reaches to `wait_until_step`. It
is used to gradually start workers in distributed settings. One example usage
would be setting `wait_until_step=int(K*log(task_id+1))` assuming that
task_id=0 is the chief.
- - -

#### `tf.train.GlobalStepWaiterHook.__init__(wait_until_step)` {#GlobalStepWaiterHook.__init__}

Create a _GlobalStepWaiterHook.

##### Args:


*  <b>`wait_until_step`</b>: an `int` shows until which global step should we wait.


- - -

#### `tf.train.GlobalStepWaiterHook.after_create_session(session)` {#GlobalStepWaiterHook.after_create_session}

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

#### `tf.train.GlobalStepWaiterHook.after_run(run_context, run_values)` {#GlobalStepWaiterHook.after_run}

Called after each call to run().

The `run_values` argument contains results of requested ops/tensors by
`before_run()`.

The `run_context` argument is the same one send to `before_run` call.
`run_context.request_stop()` can be called to stop the iteration.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.
*  <b>`run_values`</b>: A SessionRunValues object.


- - -

#### `tf.train.GlobalStepWaiterHook.before_run(run_context)` {#GlobalStepWaiterHook.before_run}




- - -

#### `tf.train.GlobalStepWaiterHook.begin()` {#GlobalStepWaiterHook.begin}




- - -

#### `tf.train.GlobalStepWaiterHook.end(session)` {#GlobalStepWaiterHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


