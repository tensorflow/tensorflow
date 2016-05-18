### `tf.string_to_hash_bucket_fast(input, num_buckets, name=None)` {#string_to_hash_bucket_fast}

Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. The strings to assing a hash bucket.
*  <b>`num_buckets`</b>: An `int` that is `>= 1`. The number of buckets.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.
  A Tensor of the same shape as the input `string_tensor`.

