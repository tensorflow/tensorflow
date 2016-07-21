### `tf.assert_proper_iterable(values)` {#assert_proper_iterable}

Static assert that values is a "proper" iterable.

`Ops` that expect iterables of `Tensor` can call this to validate input.
Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.

##### Args:


*  <b>`values`</b>: Object to be checked.

##### Raises:


*  <b>`TypeError`</b>: If `values` is not iterable or is one of
    `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.

