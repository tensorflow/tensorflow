### `tf.decode_base64(input, name=None)` {#decode_base64}

Decode web-safe base64-encoded strings.

Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and _ instead of + and /.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. Base64 strings to decode.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. Decoded strings.

