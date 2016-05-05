### `tf.gather_nd(params, indices, name=None)` {#gather_nd}

Gather values from `params` according to `indices`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_N, R]` where `R` is the rank of `params`.
The innermost dimension of `indices` (with length `R`) corresponds to the
indices of `params`.

Produces an output tensor with shape `[d_0, ..., d_{n-1}]` where:

    output[i, j, k, ...] = params[indices[i, j, k, ..., :]]

e.g. for `indices` a matrix:

    output[i] = params[indices[i, :]]

##### Args:


*  <b>`params`</b>: A `Tensor`. R-D.  The tensor from which to gather values.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    (N+1)-D.  Index tensor having shape `[d_0, ..., d_N, R]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `params`.
  N-D.  Values from `params` gathered from indices given by `indices`.

