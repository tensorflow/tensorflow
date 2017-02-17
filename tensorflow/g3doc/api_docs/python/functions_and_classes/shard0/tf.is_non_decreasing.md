### `tf.is_non_decreasing(x, name=None)` {#is_non_decreasing}

Returns `True` if `x` is non-decreasing.

Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
If `x` has less than two elements, it is trivially non-decreasing.

See also:  `is_strictly_increasing`

##### Args:


*  <b>`x`</b>: Numeric `Tensor`.
*  <b>`name`</b>: A name for this operation (optional).  Defaults to "is_non_decreasing"

##### Returns:

  Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

##### Raises:


*  <b>`TypeError`</b>: if `x` is not a numeric tensor.

