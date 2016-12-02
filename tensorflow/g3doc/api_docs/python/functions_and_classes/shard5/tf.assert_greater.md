### `tf.assert_greater(x, y, data=None, summarize=None, message=None, name=None)` {#assert_greater}

Assert the condition `x > y` holds element-wise.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_greater(x, y)]):
  output = tf.reduce_sum(x)
```

This condition holds if for every pair of (possibly broadcast) elements
`x[i]`, `y[i]`, we have `x[i] > y[i]`.
If both `x` and `y` are empty, this is trivially satisfied.

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`y`</b>: Numeric `Tensor`, same dtype as and broadcastable to `x`.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`, `y`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`message`</b>: A string to prefix to the default message.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_greater".

##### Returns:

  Op that raises `InvalidArgumentError` if `x > y` is False.

