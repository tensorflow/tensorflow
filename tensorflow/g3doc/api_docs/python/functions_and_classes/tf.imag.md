### `tf.imag(in_, name=None)` {#imag}

Returns the imaginary part of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the imaginary part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(in) ==> [4.75, 5.75]
```

##### Args:


*  <b>`in_`</b>: A `Tensor` of type `complex64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.

