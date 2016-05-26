### `tf.Assert(condition, data, summarize=None, name=None)` {#Assert}

Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

NOTE: To ensure that Assert executes, one usually attaches a dependency:

```python
 # Ensure maximum element of x is smaller or equal to 1
assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
x = tf.with_dependencies([assert_op], x)
```

##### Args:


*  <b>`condition`</b>: The condition to evaluate.
*  <b>`data`</b>: The tensors to print out when condition is false.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`assert_op`</b>: An `Operation` that, when executed, raises a
  `tf.errors.InvalidArgumentError` if `condition` is not true.

