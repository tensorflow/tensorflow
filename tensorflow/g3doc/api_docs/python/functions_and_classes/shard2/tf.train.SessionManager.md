Training helper that restores from checkpoint and creates session.

This class is a small wrapper that takes care of session creation and
checkpoint recovery. It also provides functions that to facilitate
coordination among multiple training threads or processes.

* Checkpointing trained variables as the training progresses.
* Initializing variables on startup, restoring them from the most recent
  checkpoint after a crash, or wait for checkpoints to become available.

### Usage:

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
  sm = SessionManager()
  sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
```

`prepare_session()` initializes or restores a model. It requires `init_op`
and `saver` as an argument.

A second process could wait for the model to be ready by doing the following:

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will wait for the model to become ready.
  sm = SessionManager()
  sess = sm.wait_for_session(master)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
```

`wait_for_session()` waits for a model to be initialized by other processes.
- - -

#### `tf.train.SessionManager.__init__(local_init_op=None, ready_op=None, graph=None, recovery_wait_secs=30)` {#SessionManager.__init__}

Creates a SessionManager.

The `local_init_op` is an `Operation` that is run always after a new session
was created. If `None`, this step is skipped.

The `ready_op` is an `Operation` used to check if the model is ready.  The
model is considered ready if that operation returns an empty string tensor.
If the operation returns non empty string tensor, the elements are
concatenated and used to indicate to the user why the model is not ready.

If `ready_op` is `None`, the model is not checked for readiness.

`recovery_wait_secs` is the number of seconds between checks that
the model is ready.  It is used by processes to wait for a model to
be initialized or restored.  Defaults to 30 seconds.

##### Args:


*  <b>`local_init_op`</b>: An `Operation` run immediately after session creation.
     Usually used to initialize tables and local variables.
*  <b>`ready_op`</b>: An `Operation` to check if the model is initialized.
*  <b>`graph`</b>: The `Graph` that the model will use.
*  <b>`recovery_wait_secs`</b>: Seconds between checks for the model to be ready.


- - -

#### `tf.train.SessionManager.prepare_session(master, init_op=None, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None, init_feed_dict=None, init_fn=None)` {#SessionManager.prepare_session}

Creates a `Session`. Makes sure the model is ready to be used.

Creates a `Session` on 'master'. If a `saver` object is passed in, and
`checkpoint_dir` points to a directory containing valid checkpoint
files, then it will try to recover the model from checkpoint. If
no checkpoint files are available, and `wait_for_checkpoint` is
`True`, then the process would check every `recovery_wait_secs`,
up to `max_wait_secs`, for recovery to succeed.

If the model cannot be recovered successfully then it is initialized by
either running the provided `init_op`, or calling the provided `init_fn`.
It is an error if the model cannot be recovered and neither an `init_op`
or an `init_fn` are passed.

This is a convenient function for the following, with a few error checks
added:

```python
sess, initialized = self.recover_session(master)
if not initialized:
  if init_op:
    sess.run(init_op, feed_dict=init_feed_dict)
  if init_fn;
    init_fn(sess)
return sess
```

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`init_op`</b>: Optional `Operation` used to initialize the model.
*  <b>`saver`</b>: A `Saver` object used to restore a model.
*  <b>`checkpoint_dir`</b>: Path to the checkpoint files.
*  <b>`wait_for_checkpoint`</b>: Whether to wait for checkpoint to become available.
*  <b>`max_wait_secs`</b>: Maximum time to wait for checkpoints to become available.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.
*  <b>`init_feed_dict`</b>: Optional dictionary that maps `Tensor` objects to feed
    values.  This feed dictionary is passed to the session `run()` call when
    running the init op.
*  <b>`init_fn`</b>: Optional callable used to initialize the model. Called after the
    optional `init_op` is called.  The callable must accept one argument,
    the session being initialized.

##### Returns:

  A `Session` object that can be used to drive the model.

##### Raises:


*  <b>`RuntimeError`</b>: If the model cannot be initialized or recovered.


- - -

#### `tf.train.SessionManager.recover_session(master, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None)` {#SessionManager.recover_session}

Creates a `Session`, recovering if possible.

Creates a new session on 'master'.  If the session is not initialized
and can be recovered from a checkpoint, recover it.

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`saver`</b>: A `Saver` object used to restore a model.
*  <b>`checkpoint_dir`</b>: Path to the checkpoint files.
*  <b>`wait_for_checkpoint`</b>: Whether to wait for checkpoint to become available.
*  <b>`max_wait_secs`</b>: Maximum time to wait for checkpoints to become available.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.

##### Returns:

  A pair (sess, initialized) where 'initialized' is `True` if
  the session could be recovered, `False` otherwise.


- - -

#### `tf.train.SessionManager.wait_for_session(master, config=None, max_wait_secs=inf)` {#SessionManager.wait_for_session}

Creates a new `Session` and waits for model to be ready.

Creates a new `Session` on 'master'.  Waits for the model to be
initialized or recovered from a checkpoint.  It's expected that
another thread or process will make the model ready, and that this
is intended to be used by threads/processes that participate in a
distributed training configuration where a different thread/process
is responsible for initializing or recovering the model being trained.

NB: The amount of time this method waits for the session is bounded
by max_wait_secs. By default, this function will wait indefinitely.

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: Optional ConfigProto proto used to configure the session.
*  <b>`max_wait_secs`</b>: Maximum time to wait for the session to become available.

##### Returns:

  A `Session`. May be None if the operation exceeds the timeout
  specified by config.operation_timeout_in_ms.

##### Raises:

  tf.DeadlineExceededError: if the session is not available after
    max_wait_secs.


