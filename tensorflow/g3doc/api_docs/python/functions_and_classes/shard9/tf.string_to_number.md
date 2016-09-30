### `tf.string_to_number(string_tensor, out_type=None, name=None)` {#string_to_number}

Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

##### Args:


*  <b>`string_tensor`</b>: A `Tensor` of type `string`.
*  <b>`out_type`</b>: An optional `tf.DType` from: `tf.float32, tf.int32`. Defaults to `tf.float32`.
    The numeric type to interpret each string in `string_tensor` as.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `out_type`.
  A Tensor of the same shape as the input `string_tensor`.

