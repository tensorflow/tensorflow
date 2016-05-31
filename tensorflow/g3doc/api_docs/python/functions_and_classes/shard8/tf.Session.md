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
    Defaults to using an in-process engine. At present, no value
    other than the empty string is supported.
*  <b>`graph`</b>: (Optional.) The `Graph` to be launched (described above).
*  <b>`config`</b>: (Optional.) A [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
    protocol buffer with configuration options for the session.


- - -

#### `tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#Session.run}

Runs the operations and evaluates the tensors in `fetches`.

This method runs one "step" of TensorFlow computation, by
running the necessary graph fragment to execute every `Operation`
and evaluate every `Tensor` in `fetches`, substituting the values in
`feed_dict` for the corresponding input values.

The `fetches` argument may be a single graph element, a list of
graph elements, or a dictionary whose values are the above. The type of
`fetches` determines the return value of this
method. A graph element can be one of the following types:

* If an element of `fetches` is an
  [`Operation`](../../api_docs/python/framework.md#Operation), the
  corresponding fetched value will be `None`.
* If an element of `fetches` is a
  [`Tensor`](../../api_docs/python/framework.md#Tensor), the corresponding
  fetched value will be a numpy ndarray containing the value of that tensor.
* If an element of `fetches` is a
  [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
  the corresponding fetched value will be a
  [`SparseTensorValue`](../../api_docs/python/sparse_ops.md#SparseTensorValue)
  containing the value of that sparse tensor.
* If an element of `fetches` is produced by a `get_tensor_handle` op,
  the corresponding fetched value will be a numpy ndarray containing the
  handle of that tensor.

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
[`Tensor.run()`](../../api_docs/python/framework.md#Tensor.run) should be
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


