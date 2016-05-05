### `tf.assert_rank(x, rank, data=None, summarize=None, name=None)` {#assert_rank}

Assert `x` has rank equal to `rank`.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`rank`</b>: Scalar `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_rank".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` has specified rank.

##### Raises:


*  <b>`ValueError`</b>: If static checks determine `x` has wrong rank.

