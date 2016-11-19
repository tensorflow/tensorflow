### `tf.contrib.framework.convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None, as_ref=False)` {#convert_to_tensor_or_sparse_tensor}

Converts value to a `SparseTensor` or `Tensor`.

##### Args:


*  <b>`value`</b>: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
    registered `Tensor` conversion function.
*  <b>`dtype`</b>: Optional element type for the returned tensor. If missing, the
    type is inferred from the type of `value`.
*  <b>`name`</b>: Optional name to use if a new `Tensor` is created.
*  <b>`as_ref`</b>: True if we want the result as a ref tensor. Only used if a new
    `Tensor` is created.

##### Returns:

  A `SparseTensor` or `Tensor` based on `value`.

##### Raises:


*  <b>`RuntimeError`</b>: If result type is incompatible with `dtype`.

