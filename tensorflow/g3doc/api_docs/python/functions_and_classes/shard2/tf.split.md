### `tf.split(axis=None, num_or_size_splits=None, value=None, name='split', split_dim=None)` {#split}

DEPRECATED: use split_v; split_v rename to split happening soon.

Splits `value` along dimension `axis` into `num_or_size_splits` smaller
tensors. Requires that `num_or_size_splits` evenly divide `value.shape[axis]`.

For example:

```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value=value, num_or_size_splits=3, axis=1)
tf.shape(split0) ==> [5, 10]
```

Note: If you are splitting along an axis by the length of that axis, consider
using unpack, e.g.

```python
num_items = t.get_shape()[axis].value
[tf.squeeze(s, [axis]) for s in
 tf.split(value=t, num_or_size_splits=num_items, axis=axis)]
```

can be rewritten as

```python
tf.unpack(t, axis=axis)
```

##### Args:


*  <b>`axis`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
    Must be in the range `[0, rank(value))`.
*  <b>`num_or_size_splits`</b>: A Python integer. The number of ways to split. Has a
    different meaning in split_v (see docs).
*  <b>`value`</b>: The `Tensor` to split.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`split_dim`</b>: The old (deprecated) name for axis.

##### Returns:

  `num_or_size_splits` `Tensor` objects resulting from splitting `value`.

