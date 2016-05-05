### `tf.batch_ifft(input, name=None)` {#batch_ifft}

Compute the inverse 1-dimensional discrete Fourier Transform over the inner-most

dimension of `input`.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `complex64`. A complex64 tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `complex64`.
  A complex64 tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its inverse 1D Fourier Transform.

