### `tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)` {#svd}

Computes the singular value decompositions of one or more matrices.

Computes the SVD of each inner matrix in `tensor` such that
`tensor[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :,
:])`

```prettyprint
# a is a tensor.
# s is a tensor of singular values.
# u is a tensor of left singular vectors.
#v is a tensor of right singular vectors.
s, u, v = svd(a)
s = svd(a, compute_uv=False)
```

##### Args:


*  <b>`tensor`</b>: `Tensor` of shape `[..., M, N]`. Let `P` be the minimum of `M` and
    `N`.
*  <b>`full_matrices`</b>: If true, compute full-sized `u` and `v`. If false
    (the default), compute only the leading `P` singular vectors.
    Ignored if `compute_uv` is `False`.
*  <b>`compute_uv`</b>: If `True` then left and right singular vectors will be
    computed and returned in `u` and `v`, respectively. Otherwise, only the
    singular values will be computed, which can be significantly faster.
*  <b>`name`</b>: string, optional name of the operation.

##### Returns:


*  <b>`s`</b>: Singular values. Shape is `[..., P]`.
*  <b>`u`</b>: Right singular vectors. If `full_matrices` is `False` (default) then
    shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
    `[..., M, M]`. Not returned if `compute_uv` is `False`.
*  <b>`v`</b>: Left singular vectors. If `full_matrices` is `False` (default) then
    shape is `[..., N, P]`. If `full_matrices` is `True` then shape is
    `[..., N, N]`. Not returned if `compute_uv` is `False`.

@compatibility(numpy)
Mostly equivalent to numpy.linalg.svd, except that the order of output
arguments here is `s`, `u`, `v` when `compute_uv` is `True`, as opposed to
`u`, `s`, `v` for numpy.linalg.svd.
@end_compatibility

