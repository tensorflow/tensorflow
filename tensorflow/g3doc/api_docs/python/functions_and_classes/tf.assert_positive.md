### `tf.assert_positive(x, data=None, summarize=None, name=None)` {#assert_positive}

Assert the condition `x > 0` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_positive(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_positive(x)], x)
```

Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
If `x` is empty this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_positive".

##### Returns:

  Op raising `InvalidArgumentError` unless `x` is all positive.

