### `tf.polygamma(a, x, name=None)` {#polygamma}

Compute the polygamma function \\(\psi^{(n)}(x)\\).

The polygamma function is defined as:

```
\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)
```
where \\(\psi(x)\\) is the digamma function.

##### Args:


*  <b>`a`</b>: A `Output`. Must be one of the following types: `float32`, `float64`.
*  <b>`x`</b>: A `Output`. Must have the same type as `a`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Output`. Has the same type as `a`.

