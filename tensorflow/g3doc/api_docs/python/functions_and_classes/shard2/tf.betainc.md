### `tf.betainc(a, b, x, name=None)` {#betainc}

Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

The regularized incomplete beta integral is defined as:

```
I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}
```
where

```
B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt
```

is the incomplete beta function and \\(B(a, b)\\) is the *complete*
beta function.

##### Args:


*  <b>`a`</b>: A `Output`. Must be one of the following types: `float32`, `float64`.
*  <b>`b`</b>: A `Output`. Must have the same type as `a`.
*  <b>`x`</b>: A `Output`. Must have the same type as `a`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Output`. Has the same type as `a`.

