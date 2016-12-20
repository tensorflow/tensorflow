Prints the given tensors once every N local steps or once every N seconds.

The tensors will be printed to the log, with `INFO` severity.
- - -

#### `tf.train.LoggingTensorHook.__init__(tensors, every_n_iter=None, every_n_secs=None)` {#LoggingTensorHook.__init__}

Initializes a LoggingHook monitor.

##### Args:


*  <b>`tensors`</b>: `dict` that maps string-valued tags to tensors/tensor names,
      or `iterable` of tensors/tensor names.
*  <b>`every_n_iter`</b>: `int`, print the values of `tensors` once every N local
      steps taken on the current worker.
*  <b>`every_n_secs`</b>: `int` or `float`, print the values of `tensors` once every N
      seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
      provided.

##### Raises:


*  <b>`ValueError`</b>: if `every_n_iter` is non-positive.


- - -

#### `tf.train.LoggingTensorHook.after_create_session(session)` {#LoggingTensorHook.after_create_session}

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

#### `tf.train.LoggingTensorHook.after_run(run_context, run_values)` {#LoggingTensorHook.after_run}




- - -

#### `tf.train.LoggingTensorHook.before_run(run_context)` {#LoggingTensorHook.before_run}




- - -

#### `tf.train.LoggingTensorHook.begin()` {#LoggingTensorHook.begin}




- - -

#### `tf.train.LoggingTensorHook.end(session)` {#LoggingTensorHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


