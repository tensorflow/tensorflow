### `tf.assert_integer(x, message=None, name=None)` {#assert_integer}

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
*  <b>`message`</b>: A string to prefix to the default message.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "assert_integer".

##### Raises:


*  <b>`TypeError`</b>: If `x.dtype` is anything other than non-quantized integer.

##### Returns:

  A `no_op` that does nothing.  Type can be determined statically.

