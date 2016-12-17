### `tf.concat(*args, **kwargs)` {#concat}

Concatenates tensors along one dimension. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-14.
Instructions for updating:
This op will be removed after the deprecation date. Please switch to tf.concat_v2().

Concatenates the list of tensors `values` along dimension `concat_dim`.  If
`values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]`, the concatenated
result has shape

    [D0, D1, ... Rconcat_dim, ...Dn]

where

    Rconcat_dim = sum(Dconcat_dim(i))

That is, the data from the input tensors is joined along the `concat_dim`
dimension.

The number of dimensions of the input tensors must match, and all dimensions
except `concat_dim` must be equal.

For example:

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
```

Note: If you are concatenating along a new axis consider using pack.
E.g.

```python
tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])
```

can be rewritten as

```python
tf.pack(tensors, axis=axis)
```

##### Args:


*  <b>`concat_dim`</b>: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
*  <b>`values`</b>: A list of `Tensor` objects or a single `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` resulting from concatenation of the input tensors.

