### `tf.conj(in_, name=None)` {#conj}

Returns the complex conjugate of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `in`. The
complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```

##### Args:


*  <b>`in_`</b>: A `Tensor` of type `complex64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `complex64`.

