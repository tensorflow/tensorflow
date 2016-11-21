### `tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)` {#setdiff1d}

Computes the difference between two lists of numbers or strings.

Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:

`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

For example, given this input:

```prettyprint
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```

This operation would return:

```prettyprint
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
```

##### Args:


*  <b>`x`</b>: A `Tensor`. 1-D. Values to keep.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
*  <b>`out_idx`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (out, idx).

*  <b>`out`</b>: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
*  <b>`idx`</b>: A `Tensor` of type `out_idx`. 1-D. Positions of `x` values preserved in `out`.

