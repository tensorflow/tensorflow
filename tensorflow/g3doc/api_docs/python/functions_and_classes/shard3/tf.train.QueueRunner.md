Holds a list of enqueue operations for a queue, each to be run in a thread.

Queues are a convenient TensorFlow mechanism to compute tensors
asynchronously using multiple threads. For example in the canonical 'Input
Reader' setup one set of threads generates filenames in a queue; a second set
of threads read records from the files, processes them, and enqueues tensors
on a second queue; a third set of threads dequeues these input records to
construct batches and runs them through training operations.

There are several delicate issues when running multiple threads that way:
closing the queues in sequence as the input is exhausted, correctly catching
and reporting exceptions, etc.

The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.
- - -

#### `tf.train.QueueRunner.__init__(queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None, queue_runner_def=None)` {#QueueRunner.__init__}

Create a QueueRunner.

On construction the `QueueRunner` adds an op to close the queue.  That op
will be run if the enqueue ops raise exceptions.

When you later call the `create_threads()` method, the `QueueRunner` will
create one thread for each op in `enqueue_ops`.  Each thread will run its
enqueue op in parallel with the other threads.  The enqueue ops do not have
to all be the same op, but it is expected that they all enqueue tensors in
`queue`.

##### Args:


*  <b>`queue`</b>: A `Queue`.
*  <b>`enqueue_ops`</b>: List of enqueue ops to run in threads later.
*  <b>`close_op`</b>: Op to close the queue. Pending enqueue ops are preserved.
*  <b>`cancel_op`</b>: Op to close the queue and cancel pending enqueue ops.
*  <b>`queue_closed_exception_types`</b>: Optional tuple of Exception types that
    indicate that the queue has been closed when raised during an enqueue
    operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common
    case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,
    when some of the enqueue ops may dequeue from other Queues.
*  <b>`queue_runner_def`</b>: Optional `QueueRunnerDef` protocol buffer. If specified,
    recreates the QueueRunner from its contents. `queue_runner_def` and the
    other arguments are mutually exclusive.

##### Raises:


*  <b>`ValueError`</b>: If both `queue_runner_def` and `queue` are both specified.
*  <b>`ValueError`</b>: If `queue` or `enqueue_ops` are not provided when not
    restoring from `queue_runner_def`.


- - -

#### `tf.train.QueueRunner.cancel_op` {#QueueRunner.cancel_op}




- - -

#### `tf.train.QueueRunner.close_op` {#QueueRunner.close_op}




- - -

#### `tf.train.QueueRunner.create_threads(sess, coord=None, daemon=False, start=False)` {#QueueRunner.create_threads}

Create threads to run the enqueue ops.

This method requires a session in which the graph was launched.  It creates
a list of threads, optionally starting them.  There is one thread for each
op passed in `enqueue_ops`.

The `coord` argument is an optional coordinator, that the threads will use
to terminate together and report exceptions.  If a coordinator is given,
this method starts an additional thread to close the queue when the
coordinator requests a stop.

This method may be called again as long as all threads from a previous call
have stopped.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`coord`</b>: Optional `Coordinator` object for reporting errors and checking
    stop conditions.
*  <b>`daemon`</b>: Boolean.  If `True` make the threads daemon threads.
*  <b>`start`</b>: Boolean.  If `True` starts the threads.  If `False` the
    caller must call the `start()` method of the returned threads.

##### Returns:

  A list of threads.

##### Raises:


*  <b>`RuntimeError`</b>: If threads from a previous call to `create_threads()` are
  still running.


- - -

#### `tf.train.QueueRunner.enqueue_ops` {#QueueRunner.enqueue_ops}




- - -

#### `tf.train.QueueRunner.exceptions_raised` {#QueueRunner.exceptions_raised}

Exceptions raised but not handled by the `QueueRunner` threads.

Exceptions raised in queue runner threads are handled in one of two ways
depending on whether or not a `Coordinator` was passed to
`create_threads()`:

* With a `Coordinator`, exceptions are reported to the coordinator and
  forgotten by the `QueueRunner`.
* Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
  made available in this `exceptions_raised` property.

##### Returns:

  A list of Python `Exception` objects.  The list is empty if no exception
  was captured.  (No exceptions are captured when using a Coordinator.)


- - -

#### `tf.train.QueueRunner.from_proto(queue_runner_def)` {#QueueRunner.from_proto}

Returns a `QueueRunner` object created from `queue_runner_def`.


- - -

#### `tf.train.QueueRunner.name` {#QueueRunner.name}

The string name of the underlying Queue.


- - -

#### `tf.train.QueueRunner.queue` {#QueueRunner.queue}




- - -

#### `tf.train.QueueRunner.queue_closed_exception_types` {#QueueRunner.queue_closed_exception_types}




- - -

#### `tf.train.QueueRunner.to_proto()` {#QueueRunner.to_proto}

Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

##### Returns:

  A `QueueRunnerDef` protocol buffer.


