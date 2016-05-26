### `tf.decode_raw(bytes, out_type, little_endian=None, name=None)` {#decode_raw}

Reinterpret the bytes of a string as a vector of numbers.

##### Args:


*  <b>`bytes`</b>: A `Tensor` of type `string`.
    All the elements must have the same length.
*  <b>`out_type`</b>: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`.
*  <b>`little_endian`</b>: An optional `bool`. Defaults to `True`.
    Whether the input `bytes` are in little-endian order.
    Ignored for `out_type` values that are stored in a single byte like
    `uint8`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `out_type`.
  A Tensor with one more dimension than the input `bytes`.  The
  added dimension will have size equal to the length of the elements
  of `bytes` divided by the number of bytes to represent `out_type`.

