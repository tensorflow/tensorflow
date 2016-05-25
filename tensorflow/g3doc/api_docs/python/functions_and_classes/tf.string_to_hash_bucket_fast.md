### `tf.string_to_hash_bucket_fast(input, num_buckets, name=None)` {#string_to_hash_bucket_fast}

Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. The strings to assign a hash bucket.
*  <b>`num_buckets`</b>: An `int` that is `>= 1`. The number of buckets.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.
  A Tensor of the same shape as the input `string_tensor`.

