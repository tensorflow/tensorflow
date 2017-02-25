Represents a sparse tensor.

TensorFlow represents a sparse tensor as three separate dense tensors:
`indices`, `values`, and `dense_shape`.  In Python, the three tensors are
collected into a `SparseTensor` class for ease of use.  If you have separate
`indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
object before passing to the ops below.

Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
comprises the following components, where `N` and `ndims` are the number
of values and number of dimensions in the `SparseTensor`, respectively:

* `indices`: A 2-D int64 tensor of dense_shape `[N, ndims]`, which specifies
  the indices of the elements in the sparse tensor that contain nonzero
  values (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]`
  specifies that the elements with indexes of [1,3] and [2,4] have
  nonzero values.

* `values`: A 1-D tensor of any type and dense_shape `[N]`, which supplies the
  values for each element in `indices`. For example, given
  `indices=[[1,3], [2,4]]`, the parameter `values=[18, 3.6]` specifies
  that element [1,3] of the sparse tensor has a value of 18, and element
  [2,4] of the tensor has a value of 3.6.

* `dense_shape`: A 1-D int64 tensor of dense_shape `[ndims]`, which specifies
  the dense_shape of the sparse tensor. Takes a list indicating the number of
  elements in each dimension. For example, `dense_shape=[3,6]` specifies a
  two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
  three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
  one-dimensional tensor with 9 elements.

The corresponding dense tensor satisfies:

```python
dense.shape = dense_shape
dense[tuple(indices[i])] = values[i]
```

By convention, `indices` should be sorted in row-major order (or equivalently
lexicographic order on the tuples `indices[i]`). This is not enforced when
`SparseTensor` objects are constructed, but most ops assume correct ordering.
If the ordering of sparse tensor `st` is wrong, a fixed version can be
obtained by calling `tf.sparse_reorder(st)`.

Example: The sparse tensor

```python
SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
```

represents the dense tensor

```python
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]
```
- - -

#### `tf.SparseTensor.__div__(sp_x, y)` {#SparseTensor.__div__}

Component-wise divides a SparseTensor by a dense Tensor.

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

##### Args:


*  <b>`sp_indices`</b>: A `Tensor` of type `int64`.
    2-D.  `N x R` matrix with the indices of non-empty values in a
    SparseTensor, possibly not in canonical ordering.
*  <b>`sp_values`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    1-D.  `N` non-empty values corresponding to `sp_indices`.
*  <b>`sp_shape`</b>: A `Tensor` of type `int64`.
    1-D.  Shape of the input SparseTensor.
*  <b>`dense`</b>: A `Tensor`. Must have the same type as `sp_values`.
    `R`-D.  The dense Tensor operand.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `sp_values`.
  1-D.  The `N` values that are operated on.


- - -

#### `tf.SparseTensor.__init__(indices, values, dense_shape)` {#SparseTensor.__init__}

Creates a `SparseTensor`.

##### Args:


*  <b>`indices`</b>: A 2-D int64 tensor of shape `[N, ndims]`.
*  <b>`values`</b>: A 1-D tensor of any type and shape `[N]`.
*  <b>`dense_shape`</b>: A 1-D int64 tensor of shape `[ndims]`.

##### Returns:

  A `SparseTensor`.


- - -

#### `tf.SparseTensor.__mul__(sp_x, y)` {#SparseTensor.__mul__}

Component-wise multiplies a SparseTensor by a dense Tensor.

The output locations corresponding to the implicitly zero elements in the sparse
tensor will be zero (i.e., will not take up storage space), regardless of the
contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

##### Args:


*  <b>`sp_indices`</b>: A `Tensor` of type `int64`.
    2-D.  `N x R` matrix with the indices of non-empty values in a
    SparseTensor, possibly not in canonical ordering.
*  <b>`sp_values`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    1-D.  `N` non-empty values corresponding to `sp_indices`.
*  <b>`sp_shape`</b>: A `Tensor` of type `int64`.
    1-D.  Shape of the input SparseTensor.
*  <b>`dense`</b>: A `Tensor`. Must have the same type as `sp_values`.
    `R`-D.  The dense Tensor operand.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `sp_values`.
  1-D.  The `N` values that are operated on.


- - -

#### `tf.SparseTensor.__str__()` {#SparseTensor.__str__}




- - -

#### `tf.SparseTensor.__truediv__(sp_x, y)` {#SparseTensor.__truediv__}

Internal helper function for 'sp_t / dense_t'.


- - -

#### `tf.SparseTensor.dense_shape` {#SparseTensor.dense_shape}

A 1-D Tensor of int64 representing the shape of the dense tensor.


- - -

#### `tf.SparseTensor.dtype` {#SparseTensor.dtype}

The `DType` of elements in this tensor.


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




- - -

#### `tf.SparseTensor.get_shape()` {#SparseTensor.get_shape}

Get the `TensorShape` representing the shape of the dense tensor.

##### Returns:

  A `TensorShape` object.


- - -

#### `tf.SparseTensor.graph` {#SparseTensor.graph}

The `Graph` that contains the index, value, and dense_shape tensors.


- - -

#### `tf.SparseTensor.indices` {#SparseTensor.indices}

The indices of non-zero values in the represented dense tensor.

##### Returns:

  A 2-D Tensor of int64 with dense_shape `[N, ndims]`, where `N` is the
    number of non-zero values in the tensor, and `ndims` is the rank.


- - -

#### `tf.SparseTensor.op` {#SparseTensor.op}

The `Operation` that produces `values` as an output.


- - -

#### `tf.SparseTensor.values` {#SparseTensor.values}

The non-zero values in the represented dense tensor.

##### Returns:

  A 1-D Tensor of any data type.


