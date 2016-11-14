### `tf.split_v(value, size_splits, split_dim=0, num=None, name='split_v')` {#split_v}

Splits a tensor into sub tensors.

If size_splits is a scalar, `num_split`, then
splits `value` along dimension `split_dim` into `num_split` smaller tensors.
Requires that `num_split` evenly divide `value.shape[split_dim]`.

If size_splits is a tensor, then
splits `value` into len(size_splits) pieces each the same size as the input
except along dimension split_dim where the size is size_splits[i].

For example:

```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split_v(1, [4, 15, 11], value)
tf.shape(split0) ==> [5, 4]
tf.shape(split1) ==> [5, 15]
tf.shape(split2) ==> [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, 3, 1)
tf.shape(split0) ==> [5, 10]
```

##### Args:


*  <b>`value`</b>: The `Tensor` to split.
*  <b>`size_splits`</b>: Either an integer indicating the number of splits along
    split_dim or a 1-D Tensor containing the sizes of each output tensor
    along split_dim. If an integer then it must evenly divide
    value.shape[split_dim]; otherwise the sum of sizes along the split
    dimension must match that of the input.
*  <b>`split_dim`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
    Must be in the range `[0, rank(value))`. Defaults to 0.
*  <b>`num`</b>: Optional, used to specify the number of outputs when it cannot be
       inferred from the shape of size_splits.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `len(size_splits)` `Tensor` objects resulting from splitting `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.

