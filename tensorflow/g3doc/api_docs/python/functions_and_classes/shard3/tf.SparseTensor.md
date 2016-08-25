Represents a sparse tensor.

TensorFlow represents a sparse tensor as three separate dense tensors:
`indices`, `values`, and `shape`.  In Python, the three tensors are
collected into a `SparseTensor` class for ease of use.  If you have separate
`indices`, `values`, and `shape` tensors, wrap them in a `SparseTensor`
object before passing to the ops below.

Concretely, the sparse tensor `SparseTensor(indices, values, shape)`
comprises the following components, where `N` and `ndims` are the number
of values and number of dimensions in the `SparseTensor`, respectively:

* `indices`: A 2-D int64 tensor of shape `[N, ndims]`, which specifies
  the indices of the elements in the sparse tensor that contain nonzero
  values (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]`
  specifies that the elements with indexes of [1,3] and [2,4] have
  nonzero values.

* `values`: A 1-D tensor of any type and shape `[N]`, which supplies the
  values for each element in `indices`. For example, given
  `indices=[[1,3], [2,4]]`, the parameter `values=[18, 3.6]` specifies
  that element [1,3] of the sparse tensor has a value of 18, and element
  [2,4] of the tensor has a value of 3.6.

* `shape`: A 1-D int64 tensor of shape `[ndims]`, which specifies the shape
  of the sparse tensor. Takes a list indicating the number of elements in
  each dimension. For example, `shape=[3,6]` specifies a two-dimensional 3x6
  tensor, `shape=[2,3,4]` specifies a three-dimensional 2x3x4 tensor, and
  `shape=[9]` specifies a one-dimensional tensor with 9 elements.

The corresponding dense tensor satisfies:

```python
dense.shape = shape
dense[tuple(indices[i])] = values[i]
```

By convention, `indices` should be sorted in row-major order (or equivalently
lexicographic order on the tuples `indices[i]`). This is not enforced when
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




