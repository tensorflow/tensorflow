### `tf.zeta(x, q, name=None)` {#zeta}

Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

The Hurwitz zeta function is defined as:

```
\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}
```

##### Args:


*  <b>`x`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`q`</b>: A `Tensor`. Must have the same type as `x`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `x`.

