### `tf.assert_integer(x, data=None, summarize=None, name=None)` {#assert_integer}

Assert that `x` is of integer dtype.

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.assert_integer(x)]):
  output = tf.reduce_sum(x)
```

Example of adding dependency to the tensor being checked:

```python
x = tf.with_dependencies([tf.assert_integer(x)], x)
```

##### Args:


*  <b>`x`</b>: `Tensor` whose basetype is integer and is not quantized.
*  <b>`data`</b>: The tensors to print out if the condition is False.  Defaults to
    error message and first few entries of `x`.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_integer".

##### Returns:

  Op that raises `InvalidArgumentError` if `x == y` is False.

