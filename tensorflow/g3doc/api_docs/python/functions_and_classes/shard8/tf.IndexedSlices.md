A sparse representation of a set of tensor slices at given indices.

This class is a simple wrapper for a pair of `Tensor` objects:

* `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
* `indices`: A 1-D integer `Tensor` with shape `[D0]`.

An `IndexedSlices` is typically used to represent a subset of a larger
tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
The values in `indices` are the indices in the first dimension of
the slices that have been extracted from the larger tensor.

The dense tensor `dense` represented by an `IndexedSlices` `slices` has

```python
dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
```

The `IndexedSlices` class is used principally in the definition of
gradients for operations that have sparse gradients
(e.g. [`tf.gather`](../../api_docs/python/array_ops.md#gather)).

Contrast this representation with
[`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
which uses multi-dimensional indices and scalar values.
- - -

#### `tf.IndexedSlices.__init__(values, indices, dense_shape=None)` {#IndexedSlices.__init__}

Creates an `IndexedSlices`.


- - -

#### `tf.IndexedSlices.__neg__()` {#IndexedSlices.__neg__}




- - -

#### `tf.IndexedSlices.__str__()` {#IndexedSlices.__str__}




- - -

#### `tf.IndexedSlices.dense_shape` {#IndexedSlices.dense_shape}

A 1-D `Tensor` containing the shape of the corresponding dense tensor.


- - -

#### `tf.IndexedSlices.device` {#IndexedSlices.device}

The name of the device on which `values` will be produced, or `None`.


- - -

#### `tf.IndexedSlices.dtype` {#IndexedSlices.dtype}

The `DType` of elements in this tensor.


- - -

#### `tf.IndexedSlices.graph` {#IndexedSlices.graph}

The `Graph` that contains the values, indices, and shape tensors.


- - -

#### `tf.IndexedSlices.indices` {#IndexedSlices.indices}

A 1-D `Tensor` containing the indices of the slices.


- - -

#### `tf.IndexedSlices.name` {#IndexedSlices.name}

The name of this `IndexedSlices`.


- - -

#### `tf.IndexedSlices.op` {#IndexedSlices.op}

The `Operation` that produces `values` as an output.


- - -

#### `tf.IndexedSlices.values` {#IndexedSlices.values}

A `Tensor` containing the values of the slices.


