### `tf.complex_abs(x, name=None)` {#complex_abs}

Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float32` or `float64` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The
absolute value is computed as \\( \sqrt{a^2 + b^2}\\).

For example:

```
# tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
tf.complex_abs(x) ==> [5.25594902, 6.60492229]
```

##### Args:


*  <b>`x`</b>: A `Tensor` of type `complex64` or `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32` or `float64`.

