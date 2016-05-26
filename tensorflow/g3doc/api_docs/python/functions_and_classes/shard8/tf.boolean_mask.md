### `tf.boolean_mask(tensor, mask, name='boolean_mask')` {#boolean_mask}

Apply boolean mask to tensor.  Numpy equivalent is `tensor[mask]`.

```python
# 1-D example
tensor = [0, 1, 2, 3]
mask = [True, False, True, False]
boolean_mask(tensor, mask) ==> [0, 2]
```

In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
the first K dimensions of `tensor`'s shape.  We then have:
  `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).

##### Args:


*  <b>`tensor`</b>: N-D tensor.
*  <b>`mask`</b>: K-D boolean tensor, K <= N and K must be known statically.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Tensor populated by entries in `tensor` corresponding to `True` values in
    `mask`.

##### Raises:


*  <b>`ValueError`</b>: If shapes do not conform.


*  <b>`Examples`</b>: 

```python
# 2-D example
tensor = [[1, 2], [3, 4], [5, 6]]
mask = [True, False, True]
boolean_mask(tensor, mask) ==> [[1, 2], [5, 6]]
```

