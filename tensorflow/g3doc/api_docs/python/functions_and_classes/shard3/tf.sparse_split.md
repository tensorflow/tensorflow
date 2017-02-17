### `tf.sparse_split(keyword_required=KeywordRequired(), sp_input=None, num_split=None, axis=None, name=None, split_dim=None)` {#sparse_split}

Split a `SparseTensor` into `num_split` tensors along `axis`.

If the `sp_input.dense_shape[axis]` is not an integer multiple of `num_split`
each slice starting from 0:`shape[axis] % num_split` gets extra one
dimension. For example, if `axis = 1` and `num_split = 2` and the
input is:

    input_tensor = shape = [2, 7]
    [    a   d e  ]
    [b c          ]

Graphically the output tensors are:

    output_tensor[0] =
    [    a ]
    [b c   ]

    output_tensor[1] =
    [ d e  ]
    [      ]

##### Args:


*  <b>`keyword_required`</b>: Python 2 standin for * (temporary for argument reorder)
*  <b>`sp_input`</b>: The `SparseTensor` to split.
*  <b>`num_split`</b>: A Python integer. The number of ways to split.
*  <b>`axis`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`split_dim`</b>: Deprecated old name for axis.

##### Returns:

  `num_split` `SparseTensor` objects resulting from splitting `value`.

##### Raises:


*  <b>`TypeError`</b>: If `sp_input` is not a `SparseTensor`.
*  <b>`ValueError`</b>: If the deprecated `split_dim` and `axis` are both non None.

