### `tf.sparse_reset_shape(sp_input, new_shape=None)` {#sparse_reset_shape}

Resets the shape of a `SparseTensor` with indices and values unchanged.

If `new_shape` is None, returns a copy of `sp_input` with its shape reset
to the tight bounding box of `sp_input`.

If `new_shape` is provided, then it must be larger or equal in all dimensions
compared to the shape of `sp_input`. When this condition is met, the returned
SparseTensor will have its shape reset to `new_shape` and its indices and
values unchanged from that of `sp_input.`

For example:

  Consider a `sp_input` with shape [2, 3, 5]:

    [0, 0, 1]: a
    [0, 1, 0]: b
    [0, 2, 2]: c
    [1, 0, 3]: d

  - It is an error to set `new_shape` as [3, 7] since this represents a
    rank-2 tensor while `sp_input` is rank-3. This is either a ValueError
    during graph construction (if both shapes are known) or an OpError during
    run time.

  - Setting `new_shape` as [2, 3, 6] will be fine as this shape is larger or
    equal in every dimension compared to the original shape [2, 3, 5].

  - On the other hand, setting new_shape as [2, 3, 4] is also an error: The
    third dimension is smaller than the original shape [2, 3, 5] (and an
    `InvalidArgumentError` will be raised).

  - If `new_shape` is None, the returned SparseTensor will have a shape
    [2, 3, 4], which is the tight bounding box of `sp_input`.

##### Args:


*  <b>`sp_input`</b>: The input `SparseTensor`.
*  <b>`new_shape`</b>: None or a vector representing the new shape for the returned
    `SparseTensor`.

##### Returns:

  A `SparseTensor` indices and values unchanged from `input_sp`. Its shape is
    `new_shape` if that is set. Otherwise it is  the tight bounding box of
     `input_sp`

##### Raises:


*  <b>`TypeError`</b>: If `sp_input` is not a `SparseTensor`.
*  <b>`ValueError`</b>: If `new_shape` represents a tensor with a different rank from
    that of `sp_input` (if shapes are known when graph is constructed).
*  <b>`OpError`</b>: 
    - If `new_shape` has dimension sizes that are too small.
    - If shapes are not known during graph construction time, and during run
      time it is found out that the ranks do not match.

