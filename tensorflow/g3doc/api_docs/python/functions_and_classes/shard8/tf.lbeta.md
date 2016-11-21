### `tf.lbeta(x, name='lbeta')` {#lbeta}

Computes `ln(|Beta(x)|)`, reducing along the last dimension.

Given one-dimensional `z = [z_0,...,z_{K-1}]`, we define

```Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)```

And for `n + 1` dimensional `x` with shape `[N1, ..., Nn, K]`, we define
`lbeta(x)[i1, ..., in] = Log(|Beta(x[i1, ..., in, :])|)`.  In other words,
the last dimension is treated as the `z` vector.

Note that if `z = [u, v]`, then
`Beta(z) = int_0^1 t^{u-1} (1 - t)^{v-1} dt`, which defines the traditional
bivariate beta function.

##### Args:


*  <b>`x`</b>: A rank `n + 1` `Tensor` with type `float`, or `double`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The logarithm of `|Beta(x)|` reducing along the last dimension.

##### Raises:


*  <b>`ValueError`</b>: If `x` is empty with rank one or less.

