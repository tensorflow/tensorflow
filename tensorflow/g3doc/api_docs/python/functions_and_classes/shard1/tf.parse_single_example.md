### `tf.parse_single_example(serialized, features, name=None, example_names=None)` {#parse_single_example}

Parses a single `Example` proto.

Similar to `parse_example`, except:

For dense tensors, the returned `Tensor` is identical to the output of
`parse_example`, except there is no batch dimension, the output shape is the
same as the shape given in `dense_shape`.

For `SparseTensor`s, the first (batch) column of the indices matrix is removed
(the indices matrix is a column vector), the values vector is unchanged, and
the first (`batch_size`) entry of the shape vector is removed (it is now a
single element vector).

One might see performance advantages by batching `Example` protos with
`parse_example` instead of using this function directly.

##### Args:


*  <b>`serialized`</b>: A scalar string Tensor, a single serialized Example.
    See `_parse_single_example_raw` documentation for more details.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_names`</b>: (Optional) A scalar string Tensor, the associated name.
    See `_parse_single_example_raw` documentation for more details.

##### Returns:

  A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

##### Raises:


*  <b>`ValueError`</b>: if any feature is invalid.

