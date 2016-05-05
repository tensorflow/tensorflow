### `tf.add_n(inputs, name=None)` {#add_n}

Add all input tensors element wise.

##### Args:


*  <b>`inputs`</b>: A list of at least 1 `Tensor` objects of the same type in: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Must all be the same size and shape.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `inputs`.

