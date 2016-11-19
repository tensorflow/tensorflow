### `tf.matching_files(pattern, name=None)` {#matching_files}

Returns the set of files matching a pattern.

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.

##### Args:


*  <b>`pattern`</b>: A `Tensor` of type `string`. A (scalar) shell wildcard pattern.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. A vector of matching filenames.

