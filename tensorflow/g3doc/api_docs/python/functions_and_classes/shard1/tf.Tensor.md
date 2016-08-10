Represents one of the outputs of an `Operation`.

*Note:* the `Tensor` class will be replaced by `Output` in the future.
Currently these two are aliases for each other.

A `Tensor` is a symbolic handle to one of the outputs of an
`Operation`. It does not hold the values of that operation's output,
but instead provides a means of computing those values in a
TensorFlow [`Session`](../../api_docs/python/client.md#Session).

This class has two primary purposes:

1. A `Tensor` can be passed as an input to another `Operation`.
   This builds a dataflow connection between operations, which
   enables TensorFlow to execute an entire `Graph` that represents a
   large, multi-step computation.

2. After the graph has been launched in a session, the value of the
   `Tensor` can be computed by passing it to
   [`Session.run()`](../../api_docs/python/client.md#Session.run).
   `t.eval()` is a shortcut for calling
   `tf.get_default_session().run(t)`.

In the following example, `c`, `d`, and `e` are symbolic `Tensor`
objects, whereas `result` is a numpy array that stores a concrete
value:

```python
# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)
```

- - -

#### `tf.Tensor.dtype` {#Tensor.dtype}

The `DType` of elements in this tensor.


- - -

#### `tf.Tensor.name` {#Tensor.name}

The string name of this tensor.


- - -

#### `tf.Tensor.value_index` {#Tensor.value_index}

The index of this tensor in the outputs of its `Operation`.


- - -

#### `tf.Tensor.graph` {#Tensor.graph}

The `Graph` that contains this tensor.


- - -

#### `tf.Tensor.op` {#Tensor.op}

The `Operation` that produces this tensor as an output.


- - -

#### `tf.Tensor.consumers()` {#Tensor.consumers}

Returns a list of `Operation`s that consume this tensor.

##### Returns:

  A list of `Operation`s.



- - -

#### `tf.Tensor.eval(feed_dict=None, session=None)` {#Tensor.eval}

Evaluates this tensor in a `Session`.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

*N.B.* Before invoking `Tensor.eval()`, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

##### Args:


*  <b>`feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    See [`Session.run()`](../../api_docs/python/client.md#Session.run) for a
    description of the valid feed values.
*  <b>`session`</b>: (Optional.) The `Session` to be used to evaluate this tensor. If
    none, the default session will be used.

##### Returns:

  A numpy array corresponding to the value of this tensor.



- - -

#### `tf.Tensor.get_shape()` {#Tensor.get_shape}

Returns the `TensorShape` that represents the shape of this tensor.

The shape is computed using shape inference functions that are
registered for each `Operation` type using `tf.RegisterShape`.
See [`TensorShape`](../../api_docs/python/framework.md#TensorShape) for more
details of what a shape represents.

The inferred shape of a tensor is used to provide shape
information without having to launch the graph in a session. This
can be used for debugging, and providing early error messages. For
example:

```python
c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(c.get_shape())
==> TensorShape([Dimension(2), Dimension(3)])

d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

print(d.get_shape())
==> TensorShape([Dimension(4), Dimension(2)])

# Raises a ValueError, because `c` and `d` do not have compatible
# inner dimensions.
e = tf.matmul(c, d)

f = tf.matmul(c, d, transpose_a=True, transpose_b=True)

print(f.get_shape())
==> TensorShape([Dimension(3), Dimension(4)])
```

In some cases, the inferred shape may have unknown dimensions. If
the caller has additional information about the values of these
dimensions, `Tensor.set_shape()` can be used to augment the
inferred shape.

##### Returns:

  A `TensorShape` representing the shape of this tensor.


- - -

#### `tf.Tensor.set_shape(shape)` {#Tensor.set_shape}

Updates the shape of this tensor.

This method can be called multiple times, and will merge the given
`shape` with the current shape of this tensor. It can be used to
provide additional information about the shape of this tensor that
cannot be inferred from the graph alone. For example, this can be used
to provide additional information about the shapes of images:

```python
_, image_data = tf.TFRecordReader(...).read(...)
image = tf.image.decode_png(image_data, channels=3)

# The height and width dimensions of `image` are data dependent, and
# cannot be computed without executing the op.
print(image.get_shape())
==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

# We know that each image in this dataset is 28 x 28 pixels.
image.set_shape([28, 28, 3])
print(image.get_shape())
==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
```

##### Args:


*  <b>`shape`</b>: A `TensorShape` representing the shape of this tensor.

##### Raises:


*  <b>`ValueError`</b>: If `shape` is not compatible with the current shape of
    this tensor.



#### Other Methods
- - -

#### `tf.Tensor.__init__(op, value_index, dtype)` {#Tensor.__init__}

Creates a new `Tensor`.

##### Args:


*  <b>`op`</b>: An `Operation`. `Operation` that computes this tensor.
*  <b>`value_index`</b>: An `int`. Index of the operation's endpoint that produces
    this tensor.
*  <b>`dtype`</b>: A `DType`. Type of elements stored in this tensor.

##### Raises:


*  <b>`TypeError`</b>: If the op is not an `Operation`.


- - -

#### `tf.Tensor.device` {#Tensor.device}

The name of the device on which this tensor will be produced, or None.


