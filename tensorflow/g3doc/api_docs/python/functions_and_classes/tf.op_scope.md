### `tf.op_scope(values, name, default_name=None)` {#op_scope}

Returns a context manager for use when defining a Python op.

This context manager validates that the given `values` are from the
same graph, ensures that graph is the default graph, and pushes a
name scope.

For example, to define a new Python op called `my_op`:

```python
def my_op(a, b, c, name=None):
  with tf.op_scope([a, b, c], name, "MyOp") as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    c = tf.convert_to_tensor(c, name="c")
    # Define some computation that uses `a`, `b`, and `c`.
    return foo_op(..., name=scope)
```

##### Args:


*  <b>`values`</b>: The list of `Tensor` arguments that are passed to the op function.
*  <b>`name`</b>: The name argument that is passed to the op function.
*  <b>`default_name`</b>: The default name to use if the `name` argument is `None`.

##### Returns:

  A context manager for use in defining Python ops. Yields the name scope.

##### Raises:


*  <b>`ValueError`</b>: if neither `name` nor `default_name` is provided.

