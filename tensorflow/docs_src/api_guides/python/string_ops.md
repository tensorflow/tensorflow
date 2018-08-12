# Strings

Note: Functions taking `Tensor` arguments can also take anything accepted by
`tf.convert_to_tensor`.

[TOC]

## Hashing

String hashing ops take a string input tensor and map each element to an
integer.

*   `tf.string_to_hash_bucket_fast`
*   `tf.string_to_hash_bucket_strong`
*   `tf.string_to_hash_bucket`

## Joining

String joining ops concatenate elements of input string tensors to produce a new
string tensor.

*   `tf.reduce_join`
*   `tf.string_join`

## Splitting

*   `tf.string_split`
*   `tf.substr`

## Conversion

*   `tf.as_string`
*   `tf.string_to_number`

*   `tf.decode_raw`
*   `tf.decode_csv`

*   `tf.encode_base64`
*   `tf.decode_base64`
