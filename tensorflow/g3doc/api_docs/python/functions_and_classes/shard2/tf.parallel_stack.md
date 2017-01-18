### `tf.parallel_stack(values, name='parallel_stack')` {#parallel_stack}

Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.

Requires that the shape of inputs be known at graph construction time.

Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the first dimension.
Given a list of length `N` of tensors of shape `(A, B, C)`; the `output`
tensor will have the shape `(N, A, B, C)`.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
parallel_stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]
```

The difference between stack and parallel_stack is that stack requires all
of the inputs be computed before the operation will begin but doesn't require
that the input shapes be known during graph construction.  Parallel stack
will copy pieces of the input into the output as they become available, in
some situations this can provide a performance benefit.

This is the opposite of unstack.  The numpy equivalent is

    tf.parallel_stack([x, y, z]) = np.asarray([x, y, z])

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects with the same shape and type.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`output`</b>: A stacked `Tensor` with the same type as `values`.

