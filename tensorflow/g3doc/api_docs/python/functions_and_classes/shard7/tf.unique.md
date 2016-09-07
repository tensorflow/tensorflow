### `tf.unique(x, out_idx=None, name=None)` {#unique}

Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

##### Args:


*  <b>`x`</b>: A `Tensor`. 1-D.
*  <b>`out_idx`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (y, idx).

*  <b>`y`</b>: A `Tensor`. Has the same type as `x`. 1-D.
*  <b>`idx`</b>: A `Tensor` of type `out_idx`. 1-D.

