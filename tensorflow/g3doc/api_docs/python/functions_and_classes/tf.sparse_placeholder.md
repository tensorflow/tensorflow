### `tf.sparse_placeholder(dtype, shape=None, name=None)` {#sparse_placeholder}

Inserts a placeholder for a sparse tensor that will be always fed.

**Important**: This sparse tensor will produce an error if evaluated.
Its value must be fed using the `feed_dict` optional argument to
`Session.run()`, `Tensor.eval()`, or `Operation.run()`.

For example:

```python
x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
  values = np.array([1.0, 2.0], dtype=np.float32)
  shape = np.array([7, 9, 2], dtype=np.int64)
  print(sess.run(y, feed_dict={
    x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.
  print(sess.run(y, feed_dict={
    x: (indices, values, shape)}))  # Will succeed.

  sp = tf.SparseTensor(indices=indices, values=values, shape=shape)
  sp_value = sp.eval(session)
  print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed.
```

##### Args:


*  <b>`dtype`</b>: The type of `values` elements in the tensor to be fed.
*  <b>`shape`</b>: The shape of the tensor to be fed (optional). If the shape is not
    specified, you can feed a sparse tensor of any shape.
*  <b>`name`</b>: A name for prefixing the operations (optional).

##### Returns:

  A `SparseTensor` that may be used as a handle for feeding a value, but not
  evaluated directly.

