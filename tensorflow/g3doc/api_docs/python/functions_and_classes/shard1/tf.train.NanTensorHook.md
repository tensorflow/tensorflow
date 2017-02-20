NaN Loss monitor.

Monitors loss and stops training if loss is NaN.
Can either fail with exception or just stop training.
- - -

#### `tf.train.NanTensorHook.__init__(loss_tensor, fail_on_nan_loss=True)` {#NanTensorHook.__init__}

Initializes NanLoss monitor.

##### Args:


*  <b>`loss_tensor`</b>: `Tensor`, the loss tensor.
*  <b>`fail_on_nan_loss`</b>: `bool`, whether to raise exception when loss is NaN.


- - -

#### `tf.train.NanTensorHook.after_create_session(session, coord)` {#NanTensorHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.
*  <b>`coord`</b>: A Coordinator object which keeps track of all threads.


- - -

#### `tf.train.NanTensorHook.after_run(run_context, run_values)` {#NanTensorHook.after_run}




- - -

#### `tf.train.NanTensorHook.before_run(run_context)` {#NanTensorHook.before_run}




- - -

#### `tf.train.NanTensorHook.begin()` {#NanTensorHook.begin}

Called once before using the session.

When called, the default graph is the one that will be launched in the
session.  The hook can modify the graph by adding new operations to it.
After the `begin()` call the graph will be finalized and the other callbacks
can not modify the graph anymore. Second call of `begin()` on the same
graph, should not change the graph.


- - -

#### `tf.train.NanTensorHook.end(session)` {#NanTensorHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


