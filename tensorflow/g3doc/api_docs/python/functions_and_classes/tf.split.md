### `tf.split(split_dim, num_split, value, name='split')` {#split}

Splits a tensor into `num_split` tensors along one dimension.

Splits `value` along dimension `split_dim` into `num_split` smaller tensors.
Requires that `num_split` evenly divide `value.shape[split_dim]`.

For example:

```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(1, 3, value)
tf.shape(split0) ==> [5, 10]
```

##### Args:


*  <b>`split_dim`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
    Must be in the range `[0, rank(value))`.
*  <b>`num_split`</b>: A Python integer. The number of ways to split.
*  <b>`value`</b>: The `Tensor` to split.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `num_split` `Tensor` objects resulting from splitting `value`.

