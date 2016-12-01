### `tf.reduce_join(inputs, reduction_indices, keep_dims=None, separator=None, name=None)` {#reduce_join}

Joins a string Tensor across the given dimensions.

Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.  Passing
an empty `reduction_indices` joins all strings in linear index order and outputs
a scalar string.


For example:

```
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> ["acbd"]
tf.reduce_join(a, [1, 0]) ==> ["abcd"]
tf.reduce_join(a, []) ==> ["abcd"]
```

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `string`.
    The input to be joined.  All reduced indices must have non-zero size.
*  <b>`reduction_indices`</b>: A `Tensor` of type `int32`.
    The dimensions to reduce over.  Dimensions are reduced in the
    order specified.  Omitting `reduction_indices` is equivalent to passing
    `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
*  <b>`keep_dims`</b>: An optional `bool`. Defaults to `False`.
    If `True`, retain reduced dimensions with length `1`.
*  <b>`separator`</b>: An optional `string`. Defaults to `""`.
    The separator to use when joining.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`.
  Has shape equal to that of the input with reduced dimensions removed or
  set to `1` depending on `keep_dims`.

