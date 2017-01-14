### `tf.meshgrid(*args, **kwargs)` {#meshgrid}

Broadcasts parameters for evaluation on an N-D grid.

Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
of N-D coordinate arrays for evaluating expressions on an N-D grid.

Notes:

`meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
When the `indexing` argument is set to 'xy' (the default), the broadcasting
instructions for the first two dimensions are swapped.

Examples:

Calling `X, Y = meshgrid(x, y)` with the tensors
```prettyprint
  x = [1, 2, 3]
  y = [4, 5, 6]
```
results in
```prettyprint
  X = [[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]]
  Y = [[4, 5, 6],
       [4, 5, 6],
       [4, 5, 6]]
```

##### Args:


*  <b>`*args`</b>: `Tensor`s with rank 1
*  <b>`indexing`</b>: Either 'xy' or 'ij' (optional, default: 'xy')
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`outputs`</b>: A list of N `Tensor`s with rank N

