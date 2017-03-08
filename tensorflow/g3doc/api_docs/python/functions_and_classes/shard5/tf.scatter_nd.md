### `tf.scatter_nd(indices, updates, shape, name=None)` {#scatter_nd}

Creates a new tensor by applying sparse `updates` to individual

values or slices within a zero tensor of the given `shape` tensor according to
indices.  This operator is the inverse of the [tf.gather_nd](#gather_nd)
operator which extracts values or slices from a given tensor.

TODO(simister): Add a link to Variable.__getitem__ documentation on slice
syntax.

`shape` is a `TensorShape` with rank `P` and `indices` is a `Tensor` of rank
`Q`.

`indices` must be integer tensor, containing indices into `shape`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `shape`.

`updates` is Tensor of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, shape[K], ..., shape[P-1]].
```

The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print sess.run(scatter)

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterNd2.png" alt>
</div>

In Python, this scatter operation would look like this:

    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print sess.run(scatter)

The resulting tensor would look like this:

    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

##### Args:


*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A Tensor. Must be one of the following types: int32, int64.
    A tensor of indices into ref.
*  <b>`updates`</b>: A `Tensor`.
    A Tensor. Must have the same type as tensor. A tensor of updated values
    to store in ref.
*  <b>`shape`</b>: A `Tensor`. Must have the same type as `indices`.
    A vector. The shape of the resulting tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `updates`.
  A new tensor with the given shape and updates applied according
  to the indices.

