### `tf.assert_rank_at_least(x, rank, data=None, summarize=None, message=None, name=None)` {#assert_rank_at_least}

Assert `x` has rank equal to `rank` or higher.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_rank_at_least(x, 2)]):
  output = tf.reduce_sum(x)
```

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`rank`</b>: Scalar `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`message`</b>: A string to prefix to the default message.
*  <b>`name`</b>: A name for this operation (optional).
    Defaults to "assert_rank_at_least".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
  If static checks determine `x` has correct rank, a `no_op` is returned.

##### Raises:


*  <b>`ValueError`</b>: If static checks determine `x` has wrong rank.

