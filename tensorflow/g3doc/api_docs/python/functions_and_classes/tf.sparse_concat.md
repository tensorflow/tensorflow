### `tf.sparse_concat(concat_dim, sp_inputs, name=None)` {#sparse_concat}

Concatenates a list of `SparseTensor` along the specified dimension.

Concatenation is with respect to the dense versions of each sparse input.
It is assumed that each inputs is a `SparseTensor` whose elements are ordered
along increasing dimension number.

All inputs' shapes must match, except for the concat dimension.  The
`indices`, `values`, and `shapes` lists must have the same length.

The output shape is identical to the inputs', except along the concat
dimension, where it is the sum of the inputs' sizes along that dimension.

The output elements will be resorted to preserve the sort order along
increasing dimension number.

This op runs in `O(M log M)` time, where `M` is the total number of non-empty
values across all inputs. This is due to the need for an internal sort in
order to concatenate efficiently across an arbitrary dimension.

For example, if `concat_dim = 1` and the inputs are

    sp_inputs[0]: shape = [2, 3]
    [0, 2]: "a"
    [1, 0]: "b"
    [1, 1]: "c"

    sp_inputs[1]: shape = [2, 4]
    [0, 1]: "d"
    [0, 2]: "e"

then the output will be

    shape = [2, 7]
    [0, 2]: "a"
    [0, 4]: "d"
    [0, 5]: "e"
    [1, 0]: "b"
    [1, 1]: "c"

Graphically this is equivalent to doing

    [    a] concat [  d e  ] = [    a   d e  ]
    [b c  ]        [       ]   [b c          ]

##### Args:


*  <b>`concat_dim`</b>: Dimension to concatenate along.
*  <b>`sp_inputs`</b>: List of `SparseTensor` to concatenate.
*  <b>`name`</b>: A name prefix for the returned tensors (optional).

##### Returns:

  A `SparseTensor` with the concatenated output.

##### Raises:


*  <b>`TypeError`</b>: If `sp_inputs` is not a list of `SparseTensor`.

