### `tf.pad(tensor, paddings, mode='CONSTANT', name=None)` {#pad}

Pads a tensor.

This operation pads a `tensor` according to the `paddings` you specify.
`paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
`tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
many values to add before the contents of `tensor` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of
`tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
`mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
no greater than `tensor.dim_size(D)`.

The padded size of each dimension D of the output is:

`paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

For example:

```python
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1,], [2, 2]].
# rank of 't' is 2.
pad(t, paddings, "CONSTANT") ==> [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 2, 3, 0, 0],
                                  [0, 0, 4, 5, 6, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]

pad(t, paddings, "REFLECT") ==> [[6, 5, 4, 5, 6, 5, 4],
                                 [3, 2, 1, 2, 3, 2, 1],
                                 [6, 5, 4, 5, 6, 5, 4],
                                 [3, 2, 1, 2, 3, 2, 1]]

pad(t, paddings, "SYMMETRIC") ==> [[2, 1, 1, 2, 3, 3, 2],
                                   [2, 1, 1, 2, 3, 3, 2],
                                   [5, 4, 4, 5, 6, 6, 5],
                                   [5, 4, 4, 5, 6, 6, 5]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`.
*  <b>`paddings`</b>: A `Tensor` of type `int32`.
*  <b>`mode`</b>: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.

##### Raises:


*  <b>`ValueError`</b>: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".

