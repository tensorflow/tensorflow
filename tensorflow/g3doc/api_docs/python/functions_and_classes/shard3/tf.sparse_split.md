### `tf.sparse_split(split_dim, num_split, sp_input, name=None)` {#sparse_split}

Split a `SparseTensor` into `num_split` tensors along `split_dim`.

If the `sp_input.shape[split_dim]` is not an integer multiple of `num_split`
each slice starting from 0:`shape[split_dim] % num_split` gets extra one
dimension. For example, if `split_dim = 1` and `num_split = 2` and the
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


*  <b>`split_dim`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
*  <b>`num_split`</b>: A Python integer. The number of ways to split.
*  <b>`sp_input`</b>: The `SparseTensor` to split.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `num_split` `SparseTensor` objects resulting from splitting `value`.

##### Raises:


*  <b>`TypeError`</b>: If `sp_input` is not a `SparseTensor`.

