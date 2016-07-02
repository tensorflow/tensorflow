A training helper that checkpoints models and computes summaries.

The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
and a `SessionManager` that takes care of common needs of Tensorflow
training programs.

#### Use for a single program

```python
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a Tensorflow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

Within the `with sv.managed_session()` block all variables in the graph have
been initialized.  In addition, a few services have been started to
checkpoint the model and add summaries to the event log.

If the program crashes and is restarted, the managed session automatically
reinitialize variables from the most recent checkpoint.

The supervisor is notified of any exception raised by one of the services.
After an exception is raised, `should_stop()` returns `True`.  In that case
the training loop should also stop.  This is why the training loop has to
check for `sv.should_stop()`.

Exceptions that indicate that the training inputs have been exhausted,
`tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
but are not re-raised from the `with` block: they indicate a normal
termination.

#### Use for multiple replicas

To train with replicas you deploy the same program in a `Cluster`.
One of the tasks must be identified as the *chief*: the task that handles
initialization, checkpoints, summaries, and recovery.  The other tasks
depend on the *chief* for these services.

The only change you have to do to the single program code is to indicate
if the program is running as the *chief*.

```python
# Choose a task as the chief. This could be based on server_def.task_index,
# or job_def.name, or job_def.tasks. It's entirely up to the end user.
# But there can be only one *chief*.
is_chief = (server_def.task_index == 0)
server = tf.train.Server(server_def)

with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that uses log directory on a shared file system.
  # Indicate if you are the 'chief'
  sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
  # Get a Session in a TensorFlow server on the cluster.
  with sv.managed_session(server.target) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

In the *chief* task, the `Supervisor` works exactly as in the first example
above.  In the other tasks `sv.managed_session()` waits for the Model to have
been intialized before returning a session to the training code.  The
non-chief tasks depend on the chief taks for initializing the model.

If one of the tasks crashes and restarts, `managed_session()`
checks if the Model is initialized.  If yes, it just creates a session and
returns it to the training code that proceeds normally.  If the model needs
to be initialized, the chief task takes care of reinitializing it; the other
tasks just wait for the model to have been initialized.

NOTE: This modified program still works fine as a single program.
The single program marks itself as the chief.

#### What `master` string to use

Whether you are running on your machine or in the cluster you can use the
following values for the --master flag:

* Specifying `''` requests an in-process session that does not use RPC.

* Specifying `'local'` requests a session that uses the RPC-based
  "Master interface" to run TensorFlow programs. See
  [`tf.train.Server.create_local_server()`](#Server.create_local_server) for
  details.

* Specifying `'grpc://hostname:port'` requests a session that uses
  the RPC interface to a specific , and also allows the in-process
  master to access remote tensorflow workers. Often, it is
  appropriate to pass `server.target` (for some `tf.train.Server`
  named `server).

#### Advanced use

##### Launching additional services

`managed_session()` launches the Checkpoint and Summary services (threads).
If you need more services to run you can simply launch them in the block
controlled by `managed_session()`.

Example: Start a thread to print losses.  We want this thread to run
every 60 seconds, so we launch it with `sv.loop()`.

  ```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess))
    while not sv.should_stop():
      sess.run(my_train_op)
  ```

##### Launching fewer services

`managed_session()` launches the "summary" and "checkpoint" threads which use
either the optionally `summary_op` and `saver` passed to the constructor, or
default ones created automatically by the supervisor.  If you want to run
your own summary and checkpointing logic, disable these services by passing
`None` to the `summary_op` and `saver` parameters.

Example: Create summaries manually every 100 steps in the chief.

  ```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
  ```

##### Custom model initialization

`managed_session()` only supports initializing the model by running an
`init_op` or restoring from the latest checkpoint.  If you have special
initialization needs, see how to specify a `local_init_op` when creating the
supervisor.  You can also use the `SessionManager` directly to create a
session and check if it could be initialized automatically.

- - -

#### `tf.train.Supervisor.__init__(graph=None, ready_op=0, is_chief=True, init_op=0, init_feed_dict=None, local_init_op=0, logdir=None, summary_op=0, saver=0, global_step=0, save_summaries_secs=120, save_model_secs=600, recovery_wait_secs=30, stop_grace_secs=120, checkpoint_basename='model.ckpt', session_manager=None, summary_writer=0, init_fn=None)` {#Supervisor.__init__}

Create a `Supervisor`.

##### Args:


*  <b>`graph`</b>: A `Graph`.  The graph that the model will use.  Defaults to the
    default `Graph`.  The supervisor may add operations to the graph before
    creating a session, but the graph should not be modified by the caller
    after passing it to the supervisor.
*  <b>`ready_op`</b>: 1-D string `Tensor`.  This tensor is evaluated by supervisors in
    `prepare_or_wait_for_session()` to check if the model is ready to use.
    The model is considered ready if it returns an empty array.  Defaults to
    the tensor returned from `tf.report_uninitialized_variables()`  If
    `None`, the model is not checked for readiness.
*  <b>`is_chief`</b>: If True, create a chief supervisor in charge of initializing
    and restoring the model.  If False, create a supervisor that relies
    on a chief supervisor for inits and restore.
*  <b>`init_op`</b>: `Operation`.  Used by chief supervisors to initialize the model
    when it can not be recovered.  Defaults to an `Operation` that
    initializes all variables.  If `None`, no initialization is done
    automatically unless you pass a value for `init_fn`, see below.
*  <b>`init_feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    This feed dictionary will be used when `init_op` is evaluated.
*  <b>`local_init_op`</b>: `Operation`. Used by all supervisors to run initializations
    that should run for every new supervisor instance. By default these
    are table initializers and initializers for local variables.
    If `None`, no further per supervisor-instance initialization is
    done automatically.
*  <b>`logdir`</b>: A string.  Optional path to a directory where to checkpoint the
    model and log events for the visualizer.  Used by chief supervisors.
    The directory will be created if it does not exist.
*  <b>`summary_op`</b>: An `Operation` that returns a Summary for the event logs.
    Used by chief supervisors if a `logdir` was specified.  Defaults to the
    operation returned from merge_all_summaries().  If `None`, summaries are
    not computed automatically.
*  <b>`saver`</b>: A Saver object.  Used by chief supervisors if a `logdir` was
    specified.  Defaults to the saved returned by Saver().
    If `None`, the model is not saved automatically.
*  <b>`global_step`</b>: An integer Tensor of size 1 that counts steps.  The value
    from 'global_step' is used in summaries and checkpoint filenames.
    Default to the op named 'global_step' in the graph if it exists, is of
    rank 1, size 1, and of type tf.int32 ot tf.int64.  If `None` the global
    step is not recorded in summaries and checkpoint files.  Used by chief
    supervisors if a `logdir` was specified.
*  <b>`save_summaries_secs`</b>: Number of seconds between the computation of
    summaries for the event log.  Defaults to 120 seconds.  Pass 0 to
    disable summaries.
*  <b>`save_model_secs`</b>: Number of seconds between the creation of model
    checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.
*  <b>`recovery_wait_secs`</b>: Number of seconds between checks that the model
    is ready.  Used by supervisors when waiting for a chief supervisor
    to initialize or restore the model.  Defaults to 30 seconds.
*  <b>`stop_grace_secs`</b>: Grace period, in seconds, given to running threads to
    stop when `stop()` is called.  Defaults to 120 seconds.
*  <b>`checkpoint_basename`</b>: The basename for checkpoint saving.
*  <b>`session_manager`</b>: `SessionManager`, which manages Session creation and
    recovery. If it is `None`, a default `SessionManager` will be created
    with the set of arguments passed in for backwards compatibility.
*  <b>`summary_writer`</b>: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None`
    to indicate that no summaries should be written.
*  <b>`init_fn`</b>: Optional callable used to initialize the model. Called
    after the optional `init_op` is called.  The callable must accept one
    argument, the session being initialized.

##### Returns:

  A `Supervisor`.


- - -

#### `tf.train.Supervisor.managed_session(master='', config=None, start_standard_services=True, close_summary_writer=True)` {#Supervisor.managed_session}

Returns a context manager for a managed session.

This context manager creates and automatically recovers a session.  It
optionally starts the standard services that handle checkpoints and
summaries.  It monitors exceptions raised from the `with` block or from the
services and stops the supervisor as needed.

The context manager is typically used as follows:

```python
def train():
  sv = tf.train.Supervisor(...)
  with sv.managed_session(<master>) as sess:
    for step in xrange(..):
      if sv.should_stop():
        break
      sess.run(<my training op>)
      ...do other things needed at each training step...
```

An exception raised from the `with` block or one of the service threads is
raised again when the block exits.  This is done after stopping all threads
and closing the session.  For example, an `AbortedError` exception, raised
in case of preemption of one of the workers in a distributed model, is
raised again when the block exits.

If you want to retry the training loop in case of preemption you can do it
as follows:

```python
def main(...):
  while True
    try:
      train()
    except tf.errors.Aborted:
      pass
```

As a special case, exceptions used for control flow, such as
`OutOfRangeError` which reports that input queues are exhausted, are not
raised again from the `with` block: they indicate a clean termination of
the training loop and are considered normal termination.

##### Args:


*  <b>`master`</b>: name of the TensorFlow master to use.  See the `tf.Session`
    constructor for how this is interpreted.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.
    Passed as-is to create the session.
*  <b>`start_standard_services`</b>: Whether to start the standard services,
    such as checkpoint, summary and step counter.
*  <b>`close_summary_writer`</b>: Whether to close the summary writer when
    closing the session.  Defaults to True.

##### Returns:

  A context manager that yields a `Session` restored from the latest
  checkpoint or initialized from scratch if not checkpoint exists.  The
  session is closed when the `with` block exits.


- - -

#### `tf.train.Supervisor.prepare_or_wait_for_session(master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)` {#Supervisor.prepare_or_wait_for_session}

Make sure the model is ready to be used.

Create a session on 'master', recovering or initializing the model as
needed, or wait for a session to be ready.  If running as the chief
and `start_standard_service` is set to True, also call the session
manager to start the standard services.

##### Args:


*  <b>`master`</b>: name of the TensorFlow master to use.  See the `tf.Session`
    constructor for how this is interpreted.
*  <b>`config`</b>: Optional ConfigProto proto used to configure the session,
    which is passed as-is to create the session.
*  <b>`wait_for_checkpoint`</b>: Whether we should wait for the availability of a
    checkpoint before creating Session. Defaults to False.
*  <b>`max_wait_secs`</b>: Maximum time to wait for the session to become available.
*  <b>`start_standard_services`</b>: Whether to start the standard services and the
    queue runners.

##### Returns:

  A Session object that can be used to drive the model.


- - -

#### `tf.train.Supervisor.start_standard_services(sess)` {#Supervisor.start_standard_services}

Start the standard services for 'sess'.

This starts services in the background.  The services started depend
on the parameters to the constructor and may include:

  - A Summary thread computing summaries every save_summaries_secs.
  - A Checkpoint thread saving the model every save_model_secs.
  - A StepCounter thread measure step time.

##### Args:


*  <b>`sess`</b>: A Session.

##### Returns:

  A list of threads that are running the standard services.  You can use
  the Supervisor's Coordinator to join these threads with:
    sv.coord.Join(<list of threads>)

##### Raises:


*  <b>`RuntimeError`</b>: If called with a non-chief Supervisor.
*  <b>`ValueError`</b>: If not `logdir` was passed to the constructor as the
    services need a log directory.


- - -

#### `tf.train.Supervisor.start_queue_runners(sess, queue_runners=None)` {#Supervisor.start_queue_runners}

Start threads for `QueueRunners`.

Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
are already started automatically when you create a session with the
supervisor, so unless you have non-collected queue runners to start
you do not need to call this explicitely.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`queue_runners`</b>: A list of `QueueRunners`. If not specified, we'll use the
    list of queue runners gathered in the graph under the key
    `GraphKeys.QUEUE_RUNNERS`.

##### Returns:

  The list of threads started for the `QueueRunners`.


- - -

#### `tf.train.Supervisor.summary_computed(sess, summary, global_step=None)` {#Supervisor.summary_computed}

Indicate that a summary was computed.

##### Args:


*  <b>`sess`</b>: A `Session` object.
*  <b>`summary`</b>: A Summary proto, or a string holding a serialized summary proto.
*  <b>`global_step`</b>: Int. global step this summary is associated with. If `None`,
    it will try to fetch the current step.

##### Raises:


*  <b>`TypeError`</b>: if 'summary' is not a Summary proto or a string.
*  <b>`RuntimeError`</b>: if the Supervisor was created without a `logdir`.



- - -

#### `tf.train.Supervisor.stop(threads=None, close_summary_writer=True)` {#Supervisor.stop}

Stop the services and the coordinator.

This does not close the session.

##### Args:


*  <b>`threads`</b>: Optional list of threads to join with the coordinator.  If
    `None`, defaults to the threads running the standard services, the
    threads started for `QueueRunners`, and the threads started by the
    `loop()` method.  To wait on additional threads, pass the
    list in this parameter.
*  <b>`close_summary_writer`</b>: Whether to close the `summary_writer`.  Defaults to
    `True` if the summary writer was created by the supervisor, `False`
    otherwise.


- - -

#### `tf.train.Supervisor.request_stop(ex=None)` {#Supervisor.request_stop}

Request that the coordinator stop the threads.

See `Coordinator.request_stop()`.

##### Args:


*  <b>`ex`</b>: Optional `Exception`, or Python `exc_info` tuple as returned by
    `sys.exc_info()`.  If this is the first call to `request_stop()` the
    corresponding exception is recorded and re-raised from `join()`.


- - -

#### `tf.train.Supervisor.should_stop()` {#Supervisor.should_stop}

Check if the coordinator was told to stop.

See `Coordinator.should_stop()`.

##### Returns:

  True if the coordinator was told to stop, False otherwise.


- - -

#### `tf.train.Supervisor.stop_on_exception()` {#Supervisor.stop_on_exception}

Context handler to stop the supervisor when an exception is raised.

See `Coordinator.stop_on_exception()`.

##### Returns:

  A context handler.


- - -

#### `tf.train.Supervisor.wait_for_stop()` {#Supervisor.wait_for_stop}

Block waiting for the coordinator to stop.



#### Other Methods
- - -

#### `tf.train.Supervisor.Loop(timer_interval_secs, target, args=None, kwargs=None)` {#Supervisor.Loop}

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
repeatedly.  Otherwise it calls it every `timer_interval_secs`
seconds.  The thread terminates when a stop is requested.

The started thread is added to the list of threads managed by the supervisor
so it does not need to be passed to the `stop()` method.

##### Args:


*  <b>`timer_interval_secs`</b>: Number. Time boundaries at which to call `target`.
*  <b>`target`</b>: A callable object.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Returns:

  The started thread.


- - -

#### `tf.train.Supervisor.PrepareSession(master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)` {#Supervisor.PrepareSession}

Make sure the model is ready to be used.

Create a session on 'master', recovering or initializing the model as
needed, or wait for a session to be ready.  If running as the chief
and `start_standard_service` is set to True, also call the session
manager to start the standard services.

##### Args:


*  <b>`master`</b>: name of the TensorFlow master to use.  See the `tf.Session`
    constructor for how this is interpreted.
*  <b>`config`</b>: Optional ConfigProto proto used to configure the session,
    which is passed as-is to create the session.
*  <b>`wait_for_checkpoint`</b>: Whether we should wait for the availability of a
    checkpoint before creating Session. Defaults to False.
*  <b>`max_wait_secs`</b>: Maximum time to wait for the session to become available.
*  <b>`start_standard_services`</b>: Whether to start the standard services and the
    queue runners.

##### Returns:

  A Session object that can be used to drive the model.


- - -

#### `tf.train.Supervisor.RequestStop(ex=None)` {#Supervisor.RequestStop}

Request that the coordinator stop the threads.

See `Coordinator.request_stop()`.

##### Args:


*  <b>`ex`</b>: Optional `Exception`, or Python `exc_info` tuple as returned by
    `sys.exc_info()`.  If this is the first call to `request_stop()` the
    corresponding exception is recorded and re-raised from `join()`.


- - -

#### `tf.train.Supervisor.ShouldStop()` {#Supervisor.ShouldStop}

Check if the coordinator was told to stop.

See `Coordinator.should_stop()`.

##### Returns:

  True if the coordinator was told to stop, False otherwise.


- - -

#### `tf.train.Supervisor.StartQueueRunners(sess, queue_runners=None)` {#Supervisor.StartQueueRunners}

Start threads for `QueueRunners`.

Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
are already started automatically when you create a session with the
supervisor, so unless you have non-collected queue runners to start
you do not need to call this explicitely.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`queue_runners`</b>: A list of `QueueRunners`. If not specified, we'll use the
    list of queue runners gathered in the graph under the key
    `GraphKeys.QUEUE_RUNNERS`.

##### Returns:

  The list of threads started for the `QueueRunners`.


- - -

#### `tf.train.Supervisor.StartStandardServices(sess)` {#Supervisor.StartStandardServices}

Start the standard services for 'sess'.

This starts services in the background.  The services started depend
on the parameters to the constructor and may include:

  - A Summary thread computing summaries every save_summaries_secs.
  - A Checkpoint thread saving the model every save_model_secs.
  - A StepCounter thread measure step time.

##### Args:


*  <b>`sess`</b>: A Session.

##### Returns:

  A list of threads that are running the standard services.  You can use
  the Supervisor's Coordinator to join these threads with:
    sv.coord.Join(<list of threads>)

##### Raises:


*  <b>`RuntimeError`</b>: If called with a non-chief Supervisor.
*  <b>`ValueError`</b>: If not `logdir` was passed to the constructor as the
    services need a log directory.


- - -

#### `tf.train.Supervisor.Stop(threads=None, close_summary_writer=True)` {#Supervisor.Stop}

Stop the services and the coordinator.

This does not close the session.

##### Args:


*  <b>`threads`</b>: Optional list of threads to join with the coordinator.  If
    `None`, defaults to the threads running the standard services, the
    threads started for `QueueRunners`, and the threads started by the
    `loop()` method.  To wait on additional threads, pass the
    list in this parameter.
*  <b>`close_summary_writer`</b>: Whether to close the `summary_writer`.  Defaults to
    `True` if the summary writer was created by the supervisor, `False`
    otherwise.


- - -

#### `tf.train.Supervisor.StopOnException()` {#Supervisor.StopOnException}

Context handler to stop the supervisor when an exception is raised.

See `Coordinator.stop_on_exception()`.

##### Returns:

  A context handler.


- - -

#### `tf.train.Supervisor.SummaryComputed(sess, summary, global_step=None)` {#Supervisor.SummaryComputed}

Indicate that a summary was computed.

##### Args:


*  <b>`sess`</b>: A `Session` object.
*  <b>`summary`</b>: A Summary proto, or a string holding a serialized summary proto.
*  <b>`global_step`</b>: Int. global step this summary is associated with. If `None`,
    it will try to fetch the current step.

##### Raises:


*  <b>`TypeError`</b>: if 'summary' is not a Summary proto or a string.
*  <b>`RuntimeError`</b>: if the Supervisor was created without a `logdir`.


- - -

#### `tf.train.Supervisor.WaitForStop()` {#Supervisor.WaitForStop}

Block waiting for the coordinator to stop.


- - -

#### `tf.train.Supervisor.coord` {#Supervisor.coord}

Return the Coordinator used by the Supervisor.

The Coordinator can be useful if you want to run multiple threads
during your training.

##### Returns:

  A Coordinator object.


- - -

#### `tf.train.Supervisor.global_step` {#Supervisor.global_step}

Return the global_step Tensor used by the supervisor.

##### Returns:

  An integer Tensor for the global_step.


- - -

#### `tf.train.Supervisor.init_feed_dict` {#Supervisor.init_feed_dict}

Return the feed dictionary used when evaluating the `init_op`.

##### Returns:

  A feed dictionary or `None`.


- - -

#### `tf.train.Supervisor.init_op` {#Supervisor.init_op}

Return the Init Op used by the supervisor.

##### Returns:

  An Op or `None`.


- - -

#### `tf.train.Supervisor.is_chief` {#Supervisor.is_chief}

Return True if this is a chief supervisor.

##### Returns:

  A bool.


- - -

#### `tf.train.Supervisor.loop(timer_interval_secs, target, args=None, kwargs=None)` {#Supervisor.loop}

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
repeatedly.  Otherwise it calls it every `timer_interval_secs`
seconds.  The thread terminates when a stop is requested.

The started thread is added to the list of threads managed by the supervisor
so it does not need to be passed to the `stop()` method.

##### Args:


*  <b>`timer_interval_secs`</b>: Number. Time boundaries at which to call `target`.
*  <b>`target`</b>: A callable object.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Returns:

  The started thread.


- - -

#### `tf.train.Supervisor.ready_op` {#Supervisor.ready_op}

Return the Ready Op used by the supervisor.

##### Returns:

  An Op or `None`.


- - -

#### `tf.train.Supervisor.save_model_secs` {#Supervisor.save_model_secs}

Return the delay between checkpoints.

##### Returns:

  A timestamp.


- - -

#### `tf.train.Supervisor.save_path` {#Supervisor.save_path}

Return the save path used by the supervisor.

##### Returns:

  A string.


- - -

#### `tf.train.Supervisor.save_summaries_secs` {#Supervisor.save_summaries_secs}

Return the delay between summary computations.

##### Returns:

  A timestamp.


- - -

#### `tf.train.Supervisor.saver` {#Supervisor.saver}

Return the Saver used by the supervisor.

##### Returns:

  A Saver object.


- - -

#### `tf.train.Supervisor.session_manager` {#Supervisor.session_manager}

Return the SessionManager used by the Supervisor.

##### Returns:

  A SessionManager object.


- - -

#### `tf.train.Supervisor.summary_op` {#Supervisor.summary_op}

Return the Summary Tensor used by the chief supervisor.

##### Returns:

  A string Tensor for the summary or `None`.


- - -

#### `tf.train.Supervisor.summary_writer` {#Supervisor.summary_writer}

Return the SummaryWriter used by the chief supervisor.

##### Returns:

  A SummaryWriter.


