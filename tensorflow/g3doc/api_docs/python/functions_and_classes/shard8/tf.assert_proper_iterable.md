### `tf.assert_proper_iterable(values)` {#assert_proper_iterable}

Static assert that values is a "proper" iterable.

`Ops` that expect iterables of `Output` can call this to validate input.
Useful since `Output`, `ndarray`, byte/text type are all iterables themselves.

##### Args:


*  <b>`values`</b>: Object to be checked.

##### Raises:


*  <b>`TypeError`</b>: If `values` is not iterable or is one of
    `Output`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.

