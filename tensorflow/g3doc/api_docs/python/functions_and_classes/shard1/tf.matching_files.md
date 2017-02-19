### `tf.matching_files(pattern, name=None)` {#matching_files}

Returns the set of files matching one or more glob patterns.

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.

##### Args:


*  <b>`pattern`</b>: A `Tensor` of type `string`.
    Shell wildcard pattern(s). Scalar or vector of type string.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. A vector of matching filenames.

