### `tf.sparse_to_indicator(sp_input, vocab_size, name=None)` {#sparse_to_indicator}

Converts a `SparseTensor` of ids into a dense bool indicator tensor.

The last dimension of `sp_input.indices` is discarded and replaced with
the values of `sp_input`.  If `sp_input.shape = [D0, D1, ..., Dn, K]`, then
`output.shape = [D0, D1, ..., Dn, vocab_size]`, where

    output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True

and False elsewhere in `output`.

For example, if `sp_input.shape = [2, 3, 4]` with non-empty values:

    [0, 0, 0]: 0
    [0, 1, 0]: 10
    [1, 0, 3]: 103
    [1, 1, 2]: 150
    [1, 1, 3]: 149
    [1, 1, 4]: 150
    [1, 2, 1]: 121

and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
tensor with False everywhere except at positions

    (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 149), (1, 1, 150),
    (1, 2, 121).

Note that repeats are allowed in the input SparseTensor.
This op is useful for converting `SparseTensor`s into dense formats for
compatibility with ops that expect dense tensors.

The input `SparseTensor` must be in row-major order.

##### Args:


*  <b>`sp_input`</b>: A `SparseTensor` with `values` property of type `int32` or
    `int64`.
*  <b>`vocab_size`</b>: A scalar int64 Tensor (or Python int) containing the new size
    of the last dimension, `all(0 <= sp_input.values < vocab_size)`.
*  <b>`name`</b>: A name prefix for the returned tensors (optional)

##### Returns:

  A dense bool indicator tensor representing the indices with specified value.

##### Raises:


*  <b>`TypeError`</b>: If `sp_input` is not a `SparseTensor`.

