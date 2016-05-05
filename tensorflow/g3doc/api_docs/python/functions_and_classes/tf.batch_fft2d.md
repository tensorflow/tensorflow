### `tf.batch_fft2d(in_, name=None)` {#batch_fft2d}

Compute the 2-dimensional discrete Fourier Transform over the inner-most

2 dimensions of `in`.

##### Args:


*  <b>`in_`</b>: A `Tensor` of type `complex64`. A complex64 tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `complex64`.
  A complex64 tensor of the same shape as `in`. The inner-most 2 dimensions
  of `in` are replaced with their 2D Fourier Transform.

