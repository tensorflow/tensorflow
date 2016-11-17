### `tf.scalar_mul(scalar, x)` {#scalar_mul}

Multiplies a scalar times an `Output` or `IndexedSlices` object.

Intended for use in gradient code which might deal with `IndexedSlices`
objects, which are easy to multiply by a scalar but more expensive to
multiply with arbitrary tensors.

##### Args:


*  <b>`scalar`</b>: A 0-D scalar `Output`. Must have known shape.
*  <b>`x`</b>: An `Output` or `IndexedSlices` to be scaled.

##### Returns:

  `scalar * x` of the same type (`Output` or `IndexedSlices`) as `x`.

##### Raises:


*  <b>`ValueError`</b>: if scalar is not a 0-D `scalar`.

