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

The [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
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

#### `tf.Session.__del__()` {#Session.__del__}




- - -

#### `tf.Session.__enter__()` {#Session.__enter__}




- - -

#### `tf.Session.__exit__(exec_type, exec_value, exec_tb)` {#Session.__exit__}




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
    Defaults to using an in-process engine. See
    [Distributed Tensorflow](https://www.tensorflow.org/how_tos/distributed/index.html)
    for more examples.
*  <b>`graph`</b>: (Optional.) The `Graph` to be launched (described above).
*  <b>`config`</b>: (Optional.) A [`ConfigProto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
    protocol buffer with configuration options for the session.


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

#### `tf.Session.graph_def` {#Session.graph_def}

A serializable version of the underlying TensorFlow graph.

##### Returns:

  A graph_pb2.GraphDef proto containing nodes for all of the Operations in
  the underlying TensorFlow graph.


- - -

#### `tf.Session.partial_run(handle, fetches, feed_dict=None)` {#Session.partial_run}

Continues the execution with more feeds and fetches.

This is EXPERIMENTAL and subject to change.

To use partial execution, a user first calls `partial_run_setup()` and
then a sequence of `partial_run()`. `partial_run_setup` specifies the
list of feeds and fetches that will be used in the subsequent
`partial_run` calls.

The optional `feed_dict` argument allows the caller to override
the value of tensors in the graph. See run() for more information.

Below is a simple example:

```python
a = array_ops.placeholder(dtypes.float32, shape=[])
b = array_ops.placeholder(dtypes.float32, shape=[])
c = array_ops.placeholder(dtypes.float32, shape=[])
r1 = math_ops.add(a, b)
r2 = math_ops.multiply(r1, c)

h = sess.partial_run_setup([r1, r2], [a, b, c])
res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
res = sess.partial_run(h, r2, feed_dict={c: res})
```

##### Args:


*  <b>`handle`</b>: A handle for a sequence of partial runs.
*  <b>`fetches`</b>: A single graph element, a list of graph elements,
    or a dictionary whose values are graph elements or lists of graph
    elements (see documentation for `run`).
*  <b>`feed_dict`</b>: A dictionary that maps graph elements to values
    (described above).

##### Returns:

  Either a single value if `fetches` is a single graph element, or
  a list of values if `fetches` is a list, or a dictionary with the
  same keys as `fetches` if that is a dictionary
  (see documentation for `run`).

##### Raises:

  tf.errors.OpError: Or one of its subclasses on error.


- - -

#### `tf.Session.partial_run_setup(fetches, feeds=None)` {#Session.partial_run_setup}

Sets up a graph with feeds and fetches for partial run.

This is EXPERIMENTAL and subject to change.

Note that contrary to `run`, `feeds` only specifies the graph elements.
The tensors will be supplied by the subsequent `partial_run` calls.

##### Args:


*  <b>`fetches`</b>: A single graph element, or a list of graph elements.
*  <b>`feeds`</b>: A single graph element, or a list of graph elements.

##### Returns:

  A handle for partial run.

##### Raises:


*  <b>`RuntimeError`</b>: If this `Session` is in an invalid state (e.g. has been
    closed).
*  <b>`TypeError`</b>: If `fetches` or `feed_dict` keys are of an inappropriate type.
  tf.errors.OpError: Or one of its subclasses if a TensorFlow error happens.


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


- - -

#### `tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#Session.run}

Runs operations and evaluates tensors in `fetches`.

This method runs one "step" of TensorFlow computation, by
running the necessary graph fragment to execute every `Operation`
and evaluate every `Tensor` in `fetches`, substituting the values in
`feed_dict` for the corresponding input values.

The `fetches` argument may be a single graph element, or an arbitrarily
nested list, tuple, namedtuple, dict, or OrderedDict containing graph
elements at its leaves.  A graph element can be one of the following types:

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

#### `tf.Session.sess_str` {#Session.sess_str}




