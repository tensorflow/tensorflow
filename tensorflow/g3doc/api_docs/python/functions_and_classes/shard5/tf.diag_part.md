### `tf.diag_part(input, name=None)` {#diag_part}

Returns the diagonal part of the tensor.

This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For example:

```prettyprint
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]

tf.diag_part(input) ==> [1, 2, 3, 4]
```

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`.
    Rank k tensor where k is 2, 4, or 6.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`. The extracted diagonal.

