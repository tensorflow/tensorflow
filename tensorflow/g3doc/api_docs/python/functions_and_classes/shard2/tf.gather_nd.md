### `tf.gather_nd(params, indices, name=None)` {#gather_nd}

Gather values or slices from `params` according to `indices`.

`params` is a Tensor of rank `R` and `indices` is a Tensor of rank `M`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_N, R]` where `0 < R <= M`.

The innermost dimension of `indices` (with length `R`) corresponds to
indices into elements (if `R = M`) or slices (if `R < M`) along the `N`th
dimension of `params`.

Produces an output tensor with shape

    [d_0, ..., d_{n-1}, params.shape[R], ..., params.shape[M-1]].

Some examples below.

Simple indexing into a matrix:

    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']

Slice indexing into a matrix:

    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]

Indexing into a 3-tensor:

    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['a1', 'b1'], ['c1', 'd1']]]


    indices = [[0, 1], [1, 0]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['c0', 'd0'], ['a1', 'b1']]


    indices = [[0, 0, 1], [1, 0, 1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = ['b0', 'b1']

Batched indexing into a matrix:

    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]

Batched slice indexing into a matrix:

    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]

Batched indexing into a 3-tensor:

    indices = [[[1]], [[0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[[['a1', 'b1'], ['c1', 'd1']]],
              [[['a0', 'b0'], ['c0', 'd0']]]]


    indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['c0', 'd0'], ['a1', 'b1']],
              [['a0', 'b0'], ['c1', 'd1']]]


    indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['b0', 'b1'], ['d0', 'c1']]

##### Args:


*  <b>`params`</b>: A `Tensor`. `M-D`.  The tensor from which to gather values.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    `(N+1)-D`.  Index tensor having shape `[d_0, ..., d_N, R]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `params`.
  `(N+M-R)-D`.  Values from `params` gathered from indices given by
  `indices`.

