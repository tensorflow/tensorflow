### `tf.as_dtype(type_value)` {#as_dtype}

Converts the given `type_value` to a `DType`.

##### Args:


*  <b>`type_value`</b>: A value that can be converted to a `tf.DType`
    object. This may currently be a `tf.DType` object, a
    [`DataType` enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
    a string type name, or a `numpy.dtype`.

##### Returns:

  A `DType` corresponding to `type_value`.

##### Raises:


*  <b>`TypeError`</b>: If `type_value` cannot be converted to a `DType`.

