### `tf.matrix_band_part(input, num_lower, num_upper, name=None)` {#matrix_band_part}

Copy a tensor setting everything outside a central band in each innermost matrix

to zero.

The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function

`in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                 (num_upper < 0 || (n-m) <= num_upper)`.

For example:

```prettyprint
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],

tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                       [-1,  0,  1, 2]
                                       [ 0, -1,  0, 1]
                                       [ 0,  0, -1, 0]],

tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                      [-1,  0,  1, 0]
                                      [-2, -1,  0, 1]
                                      [ 0, -2, -1, 0]]
```

Useful special cases:

```prettyprint
 tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.matrix_band_part(input, 0, 0) ==> Diagonal.
```

##### Args:


*  <b>`input`</b>: A `Tensor`. Rank `k` tensor.
*  <b>`num_lower`</b>: A `Tensor` of type `int64`.
    0-D tensor. Number of subdiagonals to keep. If negative, keep entire
    lower triangle.
*  <b>`num_upper`</b>: A `Tensor` of type `int64`.
    0-D tensor. Number of superdiagonals to keep. If negative, keep
    entire upper triangle.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Rank `k` tensor of the same shape as input. The extracted banded tensor.

