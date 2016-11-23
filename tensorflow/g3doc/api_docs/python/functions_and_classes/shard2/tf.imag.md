### `tf.imag(input, name=None)` {#imag}

Returns the imaginary part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float32` or `float64` that is the imaginary part of each element in
`input`. All elements in `input` must be complex numbers of the form \(a +
bj\), where *a* is the real part and *b* is the imaginary part returned by
this operation.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(input) ==> [4.75, 5.75]
```

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `complex64`,
    `complex128`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32` or `float64`.

