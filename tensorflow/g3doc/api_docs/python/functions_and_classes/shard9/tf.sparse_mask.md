### `tf.sparse_mask(a, mask_indices, name=None)` {#sparse_mask}

Masks elements of `IndexedSlices`.

Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
contains a subset of the slices of `a`. Only the slices at indices not
specified in `mask_indices` are returned.

This is useful when you need to extract a subset of slices in an
`IndexedSlices` object.

For example:

```python
# `a` contains slices at indices [12, 26, 37, 45] from a large tensor
# with shape [1000, 10]
a.indices => [12, 26, 37, 45]
tf.shape(a.values) => [4, 10]

# `b` will be the subset of `a` slices at its second and third indices, so
# we want to mask its first and last indices (which are at absolute
# indices 12, 45)
b = tf.sparse_mask(a, [12, 45])

b.indices => [26, 37]
tf.shape(b.values) => [2, 10]

```

##### Args:


*  <b>`a`</b>: An `IndexedSlices` instance.
*  <b>`mask_indices`</b>: Indices of elements to mask.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The masked `IndexedSlices` instance.

