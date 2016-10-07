### `tf.parse_tensor(serialized, out_type, name=None)` {#parse_tensor}

Transforms a serialized tensorflow.TensorProto proto into a Tensor.

##### Args:


*  <b>`serialized`</b>: A `Tensor` of type `string`.
    A scalar string containing a serialized TensorProto proto.
*  <b>`out_type`</b>: A `tf.DType`.
    The type of the serialized tensor.  The provided type must match the
    type of the serialized tensor and no implicit conversion will take place.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `out_type`. A Tensor of type `out_type`.

