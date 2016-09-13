<!-- This file is machine generated: DO NOT EDIT! -->

# Running Graphs
[TOC]

This library contains classes for launching graphs and executing operations.

The [basic usage](../../get_started/index.md#basic-usage) guide has
examples of how a graph is launched in a [`tf.Session`](#Session).

## Session management

- - -

### `class tf.Session` {#Session}

A class for running TensorFlow operations.

A `Session` object encapsulates the environment in which `Operation`
objects are executed, and `Tensor` objects are evaluated. For
example:

```python
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(c))
```

A session may own resources, such as
[variables](../../api_docs/python/state_ops.md#Variable), [queues](../../api_docs/python/io_ops.md#QueueBase),
and [readers](../../api_docs/python/io_ops.md#ReaderBase). It is important to release
these resources when they are no longer required. To do this, either
invoke the [`close()`](#Session.close) method on the session, or use
the session as a context manager. The following two examples are
equivalent:

```python
# Using the `close()` method.
sess = tf.Session()
sess.run(...)
sess.close()

# Using the context manager.
with tf.Session() as sess:
  sess.run(...)
```

The [`ConfigProto`]
(https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
protocol buffer exposes various configuration options for a
session. For example, to create a session that uses soft constraints
for device placement, and log the resulting placement decisions,
create a session as follows:

```python
# Launch the graph in a session that allows soft device placement and
# logs the placement decisions.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))
```

- - -

#### `tf.Session.__init__(target='', graph=None, config=None)` {#Session.__init__}

Creates a new TensorFlow session.

If no `graph` argument is specified when constructing the session,
the default graph will be launched in the session. If you are
using more than one graph (created with `tf.Graph()` in the same
process, you will have to use different sessions for each graph,
but each graph can be used in multiple sessions. In this case, it
is often clearer to pass the graph to be launched explicitly to
the session constructor.

##### Args:


*  <b>`target`</b>: (Optional.) The execution engine to connect to.
    Defaults to using an in-process engine. See [Distributed Tensorflow]
    (https://www.tensorflow.org/how_tos/distributed/index.html)
    for more examples.
*  <b>`graph`</b>: (Optional.) The `Graph` to be launched (described above).
*  <b>`config`</b>: (Optional.) A [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
    protocol buffer with configuration options for the session.


- - -

#### `tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#Session.run}

Runs operations and evaluates tensors in `fetches`.

This method runs one "step" of TensorFlow computation, by
running the necessary graph fragment to execute every `Operation`
and evaluate every `Tensor` in `fetches`, substituting the values in
`feed_dict` for the corresponding input values.

The `fetches` argument may be a single graph element, or an arbitrarily
nested list, tuple, namedtuple, or dict containing graph elements at its
leaves.  A graph element can be one of the following types:

* An [`Operation`](../../api_docs/python/framework.md#Operation).
  The corresponding fetched value will be `None`.
* A [`Tensor`](../../api_docs/python/framework.md#Tensor).
  The corresponding fetched value will be a numpy ndarray containing the
  value of that tensor.
* A [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor).
  The corresponding fetched value will be a
  [`SparseTensorValue`](../../api_docs/python/sparse_ops.md#SparseTensorValue)
  containing the value of that sparse tensor.
* A `get_tensor_handle` op.  The corresponding fetched value will be a
  numpy ndarray containing the handle of that tensor.
* A `string` which is the name of a tensor or operation in the graph.

The value returned by `run()` has the same shape as the `fetches` argument,
where the leaves are replaced by the corresponding values returned by
TensorFlow.

Example:

```python
   a = tf.constant([10, 20])
   b = tf.constant([1.0, 2.0])
   # 'fetches' can be a singleton
   v = session.run(a)
   # v is the numpy array [10, 20]
   # 'fetches' can be a list.
   v = session.run([a, b])
   # v a Python list with 2 numpy arrays: the numpy array [10, 20] and the
   # 1-D array [1.0, 2.0]
   # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
   MyData = collections.namedtuple('MyData', ['a', 'b'])
   v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
   # v is a dict with
   # v['k1'] is a MyData namedtuple with 'a' the numpy array [10, 20] and
   # 'b' the numpy array [1.0, 2.0]
   # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
   # [10, 20].
```

The optional `feed_dict` argument allows the caller to override
the value of tensors in the graph. Each key in `feed_dict` can be
one of the following types:

* If the key is a [`Tensor`](../../api_docs/python/framework.md#Tensor), the
  value may be a Python scalar, string, list, or numpy ndarray
  that can be converted to the same `dtype` as that
  tensor. Additionally, if the key is a
  [placeholder](../../api_docs/python/io_ops.md#placeholder), the shape of
  the value will be checked for compatibility with the placeholder.
* If the key is a
  [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
  the value should be a
  [`SparseTensorValue`](../../api_docs/python/sparse_ops.md#SparseTensorValue).
* If the key is a nested tuple of `Tensor`s or `SparseTensor`s, the value
  should be a nested tuple with the same structure that maps to their
  corresponding values as above.

Each value in `feed_dict` must be convertible to a numpy array of the dtype
of the corresponding key.

The optional `options` argument expects a [`RunOptions`] proto. The options
allow controlling the behavior of this particular step (e.g. turning tracing
on).

The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
appropriate, the non-Tensor output of this step will be collected there. For
example, when users turn on tracing in `options`, the profiled info will be
collected into this argument and passed back.

##### Args:


*  <b>`fetches`</b>: A single graph element, a list of graph elements,
    or a dictionary whose values are graph elements or lists of graph
    elements (described above).
*  <b>`feed_dict`</b>: A dictionary that maps graph elements to values
    (described above).
*  <b>`options`</b>: A [`RunOptions`] protocol buffer
*  <b>`run_metadata`</b>: A [`RunMetadata`] protocol buffer

##### Returns:

  Either a single value if `fetches` is a single graph element, or
  a list of values if `fetches` is a list, or a dictionary with the
  same keys as `fetches` if that is a dictionary (described above).

##### Raises:


*  <b>`RuntimeError`</b>: If this `Session` is in an invalid state (e.g. has been
    closed).
*  <b>`TypeError`</b>: If `fetches` or `feed_dict` keys are of an inappropriate type.
*  <b>`ValueError`</b>: If `fetches` or `feed_dict` keys are invalid or refer to a
    `Tensor` that doesn't exist.


- - -

#### `tf.Session.close()` {#Session.close}

Closes this session.

Calling this method frees all resources associated with the session.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    closing the TensorFlow session.



- - -

#### `tf.Session.graph` {#Session.graph}

The graph that was launched in this session.



- - -

#### `tf.Session.as_default()` {#Session.as_default}

Returns a context manager that makes this object the default session.

Use with the `with` keyword to specify that calls to
[`Operation.run()`](../../api_docs/python/framework.md#Operation.run) or
[`Tensor.eval()`](../../api_docs/python/framework.md#Tensor.eval) should be
executed in this session.

```python
c = tf.constant(..)
sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  print(c.eval())
```

To get the current default session, use
[`tf.get_default_session()`](#get_default_session).


*N.B.* The `as_default` context manager *does not* close the
session when you exit the context, and you must close the session
explicitly.

```python
c = tf.constant(...)
sess = tf.Session()
with sess.as_default():
  print(c.eval())
# ...
with sess.as_default():
  print(c.eval())

sess.close()
```

Alternatively, you can use `with tf.Session():` to create a
session that is automatically closed on exiting the context,
including when an uncaught exception is raised.

*N.B.* The default graph is a property of the current thread. If you
create a new thread, and wish to use the default session in that
thread, you must explicitly add a `with sess.as_default():` in that
thread's function.

##### Returns:

  A context manager using this session as the default session.



- - -

#### `tf.Session.reset(target, containers=None, config=None)` {#Session.reset}

Resets resource containers on `target`, and close all connected sessions.

A resource container is distributed across all workers in the
same cluster as `target`.  When a resource container on `target`
is reset, resources associated with that container will be cleared.
In particular, all Variables in the container will become undefined:
they lose their values and shapes.

NOTE:
(i) reset() is currently only implemented for distributed sessions.
(ii) Any sessions on the master named by `target` will be closed.

If no resource containers are provided, all containers are reset.

##### Args:


*  <b>`target`</b>: The execution engine to connect to.
*  <b>`containers`</b>: A list of resource container name strings, or `None` if all of
    all the containers are to be reset.
*  <b>`config`</b>: (Optional.) Protocol buffer with configuration options.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    resetting containers.



#### Other Methods
- - -

#### `tf.Session.__enter__()` {#Session.__enter__}




- - -

#### `tf.Session.__exit__(exec_type, exec_value, exec_tb)` {#Session.__exit__}





- - -

### `class tf.InteractiveSession` {#InteractiveSession}

A TensorFlow `Session` for use in interactive contexts, such as a shell.

The only difference with a regular `Session` is that an `InteractiveSession`
installs itself as the default session on construction.
The methods [`Tensor.eval()`](../../api_docs/python/framework.md#Tensor.eval)
and [`Operation.run()`](../../api_docs/python/framework.md#Operation.run)
will use that session to run ops.

This is convenient in interactive shells and [IPython
notebooks](http://ipython.org), as it avoids having to pass an explicit
`Session` object to run ops.

For example:

```python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close()
```

Note that a regular session installs itself as the default session when it
is created in a `with` statement.  The common usage in non-interactive
programs is to follow that pattern:

```python
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
  # We can also use 'c.eval()' here.
  print(c.eval())
```

- - -

#### `tf.InteractiveSession.__init__(target='', graph=None, config=None)` {#InteractiveSession.__init__}

Creates a new interactive TensorFlow session.

If no `graph` argument is specified when constructing the session,
the default graph will be launched in the session. If you are
using more than one graph (created with `tf.Graph()` in the same
process, you will have to use different sessions for each graph,
but each graph can be used in multiple sessions. In this case, it
is often clearer to pass the graph to be launched explicitly to
the session constructor.

##### Args:


*  <b>`target`</b>: (Optional.) The execution engine to connect to.
    Defaults to using an in-process engine.
*  <b>`graph`</b>: (Optional.) The `Graph` to be launched (described above).
*  <b>`config`</b>: (Optional) `ConfigProto` proto used to configure the session.


- - -

#### `tf.InteractiveSession.close()` {#InteractiveSession.close}

Closes an `InteractiveSession`.




- - -

### `tf.get_default_session()` {#get_default_session}

Returns the default session for the current thread.

The returned `Session` will be the innermost session on which a
`Session` or `Session.as_default()` context has been entered.

NOTE: The default session is a property of the current thread. If you
create a new thread, and wish to use the default session in that
thread, you must explicitly add a `with sess.as_default():` in that
thread's function.

##### Returns:

  The default `Session` being used in the current thread.



## Error classes

- - -

### `class tf.OpError` {#OpError}

A generic error that is raised when TensorFlow execution fails.

Whenever possible, the session will raise a more specific subclass
of `OpError` from the `tf.errors` module.

- - -

#### `tf.OpError.op` {#OpError.op}

The operation that failed, if known.

*N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
or `Recv` op, there will be no corresponding
[`Operation`](../../api_docs/python/framework.md#Operation)
object.  In that case, this will return `None`, and you should
instead use the [`OpError.node_def`](#OpError.node_def) to
discover information about the op.

##### Returns:

  The `Operation` that failed, or None.


- - -

#### `tf.OpError.node_def` {#OpError.node_def}

The `NodeDef` proto representing the op that failed.



#### Other Methods
- - -

#### `tf.OpError.__init__(node_def, op, message, error_code)` {#OpError.__init__}

Creates a new `OpError` indicating that a particular op failed.

##### Args:


*  <b>`node_def`</b>: The `node_def_pb2.NodeDef` proto representing the op that
    failed, if known; otherwise None.
*  <b>`op`</b>: The `ops.Operation` that failed, if known; otherwise None.
*  <b>`message`</b>: The message string describing the failure.
*  <b>`error_code`</b>: The `error_codes_pb2.Code` describing the error.


- - -

#### `tf.OpError.__str__()` {#OpError.__str__}




- - -

#### `tf.OpError.error_code` {#OpError.error_code}

The integer error code that describes the error.


- - -

#### `tf.OpError.message` {#OpError.message}

The error message that describes the error.



- - -

### `class tf.errors.CancelledError` {#CancelledError}

Raised when an operation or step is cancelled.

For example, a long-running operation (e.g.
[`queue.enqueue()`](../../api_docs/python/io_ops.md#QueueBase.enqueue) may be
cancelled by running another operation (e.g.
[`queue.close(cancel_pending_enqueues=True)`](../../api_docs/python/io_ops.md#QueueBase.close),
or by [closing the session](../../api_docs/python/client.md#Session.close).
A step that is running such a long-running operation will fail by raising
`CancelledError`.

- - -

#### `tf.errors.CancelledError.__init__(node_def, op, message)` {#CancelledError.__init__}

Creates a `CancelledError`.



- - -

### `class tf.errors.UnknownError` {#UnknownError}

Unknown error.

An example of where this error may be returned is if a Status value
received from another address space belongs to an error-space that
is not known to this address space. Also errors raised by APIs that
do not return enough error information may be converted to this
error.

- - -

#### `tf.errors.UnknownError.__init__(node_def, op, message, error_code=2)` {#UnknownError.__init__}

Creates an `UnknownError`.



- - -

### `class tf.errors.InvalidArgumentError` {#InvalidArgumentError}

Raised when an operation receives an invalid argument.

This may occur, for example, if an operation is receives an input
tensor that has an invalid value or shape. For example, the
[`tf.matmul()`](../../api_docs/python/math_ops.md#matmul) op will raise this
error if it receives an input that is not a matrix, and the
[`tf.reshape()`](../../api_docs/python/array_ops.md#reshape) op will raise
this error if the new shape does not match the number of elements in the input
tensor.

- - -

#### `tf.errors.InvalidArgumentError.__init__(node_def, op, message)` {#InvalidArgumentError.__init__}

Creates an `InvalidArgumentError`.



- - -

### `class tf.errors.DeadlineExceededError` {#DeadlineExceededError}

Raised when a deadline expires before an operation could complete.

This exception is not currently used.

- - -

#### `tf.errors.DeadlineExceededError.__init__(node_def, op, message)` {#DeadlineExceededError.__init__}

Creates a `DeadlineExceededError`.



- - -

### `class tf.errors.NotFoundError` {#NotFoundError}

Raised when a requested entity (e.g., a file or directory) was not found.

For example, running the
[`tf.WholeFileReader.read()`](../../api_docs/python/io_ops.md#WholeFileReader)
operation could raise `NotFoundError` if it receives the name of a file that
does not exist.

- - -

#### `tf.errors.NotFoundError.__init__(node_def, op, message)` {#NotFoundError.__init__}

Creates a `NotFoundError`.



- - -

### `class tf.errors.AlreadyExistsError` {#AlreadyExistsError}

Raised when an entity that we attempted to create already exists.

For example, running an operation that saves a file
(e.g. [`tf.train.Saver.save()`](../../api_docs/python/train.md#Saver.save))
could potentially raise this exception if an explicit filename for an
existing file was passed.

- - -

#### `tf.errors.AlreadyExistsError.__init__(node_def, op, message)` {#AlreadyExistsError.__init__}

Creates an `AlreadyExistsError`.



- - -

### `class tf.errors.PermissionDeniedError` {#PermissionDeniedError}

Raised when the caller does not have permission to run an operation.

For example, running the
[`tf.WholeFileReader.read()`](../../api_docs/python/io_ops.md#WholeFileReader)
operation could raise `PermissionDeniedError` if it receives the name of a
file for which the user does not have the read file permission.

- - -

#### `tf.errors.PermissionDeniedError.__init__(node_def, op, message)` {#PermissionDeniedError.__init__}

Creates a `PermissionDeniedError`.



- - -

### `class tf.errors.UnauthenticatedError` {#UnauthenticatedError}

The request does not have valid authentication credentials.

This exception is not currently used.

- - -

#### `tf.errors.UnauthenticatedError.__init__(node_def, op, message)` {#UnauthenticatedError.__init__}

Creates an `UnauthenticatedError`.



- - -

### `class tf.errors.ResourceExhaustedError` {#ResourceExhaustedError}

Some resource has been exhausted.

For example, this error might be raised if a per-user quota is
exhausted, or perhaps the entire file system is out of space.

- - -

#### `tf.errors.ResourceExhaustedError.__init__(node_def, op, message)` {#ResourceExhaustedError.__init__}

Creates a `ResourceExhaustedError`.



- - -

### `class tf.errors.FailedPreconditionError` {#FailedPreconditionError}

Operation was rejected because the system is not in a state to execute it.

This exception is most commonly raised when running an operation
that reads a [`tf.Variable`](../../api_docs/python/state_ops.md#Variable)
before it has been initialized.

- - -

#### `tf.errors.FailedPreconditionError.__init__(node_def, op, message)` {#FailedPreconditionError.__init__}

Creates a `FailedPreconditionError`.



- - -

### `class tf.errors.AbortedError` {#AbortedError}

The operation was aborted, typically due to a concurrent action.

For example, running a
[`queue.enqueue()`](../../api_docs/python/io_ops.md#QueueBase.enqueue)
operation may raise `AbortedError` if a
[`queue.close()`](../../api_docs/python/io_ops.md#QueueBase.close) operation
previously ran.

- - -

#### `tf.errors.AbortedError.__init__(node_def, op, message)` {#AbortedError.__init__}

Creates an `AbortedError`.



- - -

### `class tf.errors.OutOfRangeError` {#OutOfRangeError}

Raised when an operation iterates past the valid input range.

This exception is raised in "end-of-file" conditions, such as when a
[`queue.dequeue()`](../../api_docs/python/io_ops.md#QueueBase.dequeue)
operation is blocked on an empty queue, and a
[`queue.close()`](../../api_docs/python/io_ops.md#QueueBase.close)
operation executes.

- - -

#### `tf.errors.OutOfRangeError.__init__(node_def, op, message)` {#OutOfRangeError.__init__}

Creates an `OutOfRangeError`.



- - -

### `class tf.errors.UnimplementedError` {#UnimplementedError}

Raised when an operation has not been implemented.

Some operations may raise this error when passed otherwise-valid
arguments that it does not currently support. For example, running
the [`tf.nn.max_pool()`](../../api_docs/python/nn.md#max_pool) operation
would raise this error if pooling was requested on the batch dimension,
because this is not yet supported.

- - -

#### `tf.errors.UnimplementedError.__init__(node_def, op, message)` {#UnimplementedError.__init__}

Creates an `UnimplementedError`.



- - -

### `class tf.errors.InternalError` {#InternalError}

Raised when the system experiences an internal error.

This exception is raised when some invariant expected by the runtime
has been broken. Catching this exception is not recommended.

- - -

#### `tf.errors.InternalError.__init__(node_def, op, message)` {#InternalError.__init__}

Creates an `InternalError`.



- - -

### `class tf.errors.UnavailableError` {#UnavailableError}

Raised when the runtime is currently unavailable.

This exception is not currently used.

- - -

#### `tf.errors.UnavailableError.__init__(node_def, op, message)` {#UnavailableError.__init__}

Creates an `UnavailableError`.



- - -

### `class tf.errors.DataLossError` {#DataLossError}

Raised when unrecoverable data loss or corruption is encountered.

For example, this may be raised by running a
[`tf.WholeFileReader.read()`](../../api_docs/python/io_ops.md#WholeFileReader)
operation, if the file is truncated while it is being read.

- - -

#### `tf.errors.DataLossError.__init__(node_def, op, message)` {#DataLossError.__init__}

Creates a `DataLossError`.



