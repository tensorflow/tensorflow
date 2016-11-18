### `tf.placeholder(dtype, shape=None, name=None)` {#placeholder}

Inserts a placeholder for a tensor that will be always fed.

**Important**: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.

For example:

```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

##### Args:


*  <b>`dtype`</b>: The type of elements in the tensor to be fed.
*  <b>`shape`</b>: The shape of the tensor to be fed (optional). If the shape is not
    specified, you can feed a tensor of any shape.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` that may be used as a handle for feeding a value, but not
  evaluated directly.

