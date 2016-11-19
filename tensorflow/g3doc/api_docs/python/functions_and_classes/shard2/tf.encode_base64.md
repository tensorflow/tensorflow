### `tf.encode_base64(input, pad=None, name=None)` {#encode_base64}

Encode strings into web-safe base64 format.

Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.

Web-safe means that the encoder uses - and _ instead of + and /.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. Strings to be encoded.
*  <b>`pad`</b>: An optional `bool`. Defaults to `False`.
    Bool whether padding is applied at the ends.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. Input strings encoded in base64.

