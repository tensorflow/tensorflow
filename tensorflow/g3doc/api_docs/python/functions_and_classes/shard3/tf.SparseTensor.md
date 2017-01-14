Represents a sparse tensor.

TensorFlow represents a sparse tensor as three separate dense tensors:
`indices`, `values`, and `shape`.  In Python, the three tensors are
collected into a `SparseTensor` class for ease of use.  If you have separate
`indices`, `values`, and `shape` tensors, wrap them in a `SparseTensor`
object before passing to the ops below.

Concretely, the sparse tensor `SparseTensor(indices, values, shape)` is

* `indices`: A 2-D int64 tensor of shape `[N, ndims]`.
* `values`: A 1-D tensor of any type and shape `[N]`.
* `shape`: A 1-D int64 tensor of shape `[ndims]`.

where `N` and `ndims` are the number of values, and number of dimensions in
the `SparseTensor` respectively.

The corresponding dense tensor satisfies

```python
dense.shape = shape
dense[tuple(indices[i])] = values[i]
```

By convention, `indices` should be sorted in row-major order (or equivalently
lexicographic order on the tuples `indices[i]`).  This is not enforced when
`SparseTensor` objects are constructed, but most ops assume correct ordering.
If the ordering of sparse tensor `st` is wrong, a fixed version can be
obtained by calling `tf.sparse_reorder(st)`.

Example: The sparse tensor

```python
SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], shape=[3, 4])
```

represents the dense tensor

```python
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]
```

- - -

#### `tf.SparseTensor.__init__(indices, values, shape)` {#SparseTensor.__init__}

Creates a `SparseTensor`.

##### Args:


*  <b>`indices`</b>: A 2-D int64 tensor of shape `[N, ndims]`.
*  <b>`values`</b>: A 1-D tensor of any type and shape `[N]`.
*  <b>`shape`</b>: A 1-D int64 tensor of shape `[ndims]`.

##### Returns:

  A `SparseTensor`


- - -

#### `tf.SparseTensor.indices` {#SparseTensor.indices}

The indices of non-zero values in the represented dense tensor.

##### Returns:

  A 2-D Tensor of int64 with shape `[N, ndims]`, where `N` is the
    number of non-zero values in the tensor, and `ndims` is the rank.


- - -

#### `tf.SparseTensor.values` {#SparseTensor.values}

The non-zero values in the represented dense tensor.

##### Returns:

  A 1-D Tensor of any data type.


- - -

#### `tf.SparseTensor.shape` {#SparseTensor.shape}

A 1-D Tensor of int64 representing the shape of the dense tensor.


- - -

#### `tf.SparseTensor.dtype` {#SparseTensor.dtype}

The `DType` of elements in this tensor.


- - -

#### `tf.SparseTensor.op` {#SparseTensor.op}

The `Operation` that produces `values` as an output.


- - -

#### `tf.SparseTensor.graph` {#SparseTensor.graph}

The `Graph` that contains the index, value, and shape tensors.



#### Other Methods
- - -

#### `tf.SparseTensor.eval(feed_dict=None, session=None)` {#SparseTensor.eval}

Evaluates this sparse tensor in a `Session`.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

*N.B.* Before invoking `SparseTensor.eval()`, its graph must have been
launched in a session, and either a default session must be
available, or `session` must be specified explicitly.

##### Args:


*  <b>`feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    See [`Session.run()`](../../api_docs/python/client.md#Session.run) for a
    description of the valid feed values.
*  <b>`session`</b>: (Optional.) The `Session` to be used to evaluate this sparse
    tensor. If none, the default session will be used.

##### Returns:

  A `SparseTensorValue` object.


- - -

#### `tf.SparseTensor.from_value(cls, sparse_tensor_value)` {#SparseTensor.from_value}




