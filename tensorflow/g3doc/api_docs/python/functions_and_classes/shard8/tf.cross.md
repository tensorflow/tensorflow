### `tf.cross(a, b, name=None)` {#cross}

Compute the pairwise cross product.

`a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently.

##### Args:


*  <b>`a`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    A tensor containing 3-element vectors.
*  <b>`b`</b>: A `Tensor`. Must have the same type as `a`.
    Another tensor, of same type and shape as `a`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `a`.
  Pairwise cross product of the vectors in `a` and `b`.

