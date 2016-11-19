### `tf.convert_to_tensor_or_indexed_slices(value, dtype=None, name=None, as_ref=False)` {#convert_to_tensor_or_indexed_slices}

Converts the given object to a `Tensor` or an `IndexedSlices`.

If `value` is an `IndexedSlices` or `SparseTensor` it is returned
unmodified. Otherwise, it is converted to a `Tensor` using
`convert_to_tensor()`.

##### Args:


*  <b>`value`</b>: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
    by `convert_to_tensor()`.
*  <b>`dtype`</b>: (Optional.) The required `DType` of the returned `Tensor` or
    `IndexedSlices`.
*  <b>`name`</b>: (Optional.) A name to use if a new `Tensor` is created.
*  <b>`as_ref`</b>: True if the caller wants the results as ref tensors.

##### Returns:

  An `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

##### Raises:


*  <b>`ValueError`</b>: If `dtype` does not match the element type of `value`.

