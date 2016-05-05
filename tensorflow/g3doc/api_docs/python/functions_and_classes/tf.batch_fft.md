### `tf.batch_fft(in_, name=None)` {#batch_fft}

Compute the 1-dimensional discrete Fourier Transform over the inner-most

dimension of `in`.

##### Args:


*  <b>`in_`</b>: A `Tensor` of type `complex64`. A complex64 tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `complex64`.
  A complex64 tensor of the same shape as `in`. The inner-most dimension of
  `in` is replaced with its 1D Fourier Transform.

