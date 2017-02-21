### `tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)` {#eye}

Construct an identity matrix, or a batch of matrices.

```python
# Construct one identity matrix.
tf.eye(2)
==> [[1., 0.],
     [0., 1.]]

# Construct a batch of 3 identity matricies, each 2 x 2.
# batch_identity[i, :, :] is a 2 x 2 identity matrix, i = 0, 1, 2.
batch_identity = tf.eye(2, batch_shape=[3])

# Construct one 2 x 3 "identity" matrix
tf.eye(2, num_columns=3)
==> [[ 1.,  0.,  0.],
     [ 0.,  1.,  0.]]
```

##### Args:


*  <b>`num_rows`</b>: Non-negative `int32` scalar `Tensor` giving the number of rows
    in each batch matrix.
*  <b>`num_columns`</b>: Optional non-negative `int32` scalar `Tensor` giving the number
    of columns in each batch matrix.  Defaults to `num_rows`.
*  <b>`batch_shape`</b>: `int32` `Tensor`.  If provided, returned `Tensor` will have
    leading batch dimensions of this shape.
*  <b>`dtype`</b>: The type of an element in the resulting `Tensor`
*  <b>`name`</b>: A name for this `Op`.  Defaults to "eye".

##### Returns:

  A `Tensor` of shape `batch_shape + [num_rows, num_columns]`

