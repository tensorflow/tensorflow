Prints given tensors every N iteration.

The tensors will be printed to the log, with `INFO` severity.
- - -

#### `tf.train.LoggingTensorHook.__init__(tensors, every_n_iter=100)` {#LoggingTensorHook.__init__}

Initializes a LoggingHook monitor.

##### Args:


*  <b>`tensors`</b>: `dict` of tag to tensors/names or
      `iterable` of tensors/names.
*  <b>`every_n_iter`</b>: `int`, print every N iteration.

##### Raises:


*  <b>`ValueError`</b>: if `every_n_iter` is non-positive.


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


