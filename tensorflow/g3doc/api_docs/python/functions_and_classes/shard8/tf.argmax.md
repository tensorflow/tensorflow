### `tf.argmax(input, dimension, name=None)` {#argmax}

Returns the index with the largest value across dimensions of a tensor.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
*  <b>`dimension`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    int32, 0 <= dimension < rank(input).  Describes which dimension
    of the input Tensor to reduce across. For vectors, use dimension = 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.

