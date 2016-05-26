### `tf.igammac(a, x, name=None)` {#igammac}

Compute the upper regularized incomplete Gamma function `Q(a, x)`.

The upper regularized incomplete Gamma function is defined as:

```
Q(a, x) = Gamma(a, x) / Gamma(x) = 1 - P(a, x)
```
where
```
Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt
```
is the upper incomplete Gama function.

Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
Gamma function.

##### Args:


*  <b>`a`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`x`</b>: A `Tensor`. Must have the same type as `a`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `a`.

