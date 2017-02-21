### `tf.slice(input_, begin, size, name=None)` {#slice}

Extracts a slice from a tensor.

This operation extracts a slice of size `size` from a tensor `input` starting
at the location specified by `begin`. The slice `size` is represented as a
tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
of `input` that you want to slice. The starting location (`begin`) for the
slice is represented as an offset in each dimension of `input`. In other
words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
want to slice from.

`begin` is zero-based; `size` is one-based. If `size[i]` is -1,
all remaining elements in dimension i are included in the
slice. In other words, this is equivalent to setting:

`size[i] = input.dim_size(i) - begin[i]`

This operation requires that:

`0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

For example:

```python
# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                            [4, 4, 4]]]
tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                           [[5, 5, 5]]]
```

##### Args:


*  <b>`input_`</b>: A `Tensor`.
*  <b>`begin`</b>: An `int32` or `int64` `Tensor`.
*  <b>`size`</b>: An `int32` or `int64` `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` the same type as `input`.

