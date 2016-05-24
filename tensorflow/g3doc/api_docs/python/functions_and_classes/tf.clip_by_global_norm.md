### `tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)` {#clip_by_global_norm}

Clips values of multiple tensors by the ratio of the sum of their norms.

Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
this operation returns a list of clipped tensors `list_clipped`
and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
if you've already computed the global norm for `t_list`, you can specify
the global norm with `use_norm`.

To perform the clipping, the values `t_list[i]` are set to:

    t_list[i] * clip_norm / max(global_norm, clip_norm)

where:

    global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
otherwise they're all shrunk by the global ratio.

Any of the entries of `t_list` that are of type `None` are ignored.

This is the correct way to perform gradient clipping (for example, see
[Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

However, it is slower than `clip_by_norm()` because all the parameters must be
ready before the clipping operation can be performed.

##### Args:


*  <b>`t_list`</b>: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
*  <b>`clip_norm`</b>: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
*  <b>`use_norm`</b>: A 0-D (scalar) `Tensor` of type `float` (optional). The global
    norm to use. If not provided, `global_norm()` is used to compute the norm.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`list_clipped`</b>: A list of `Tensors` of the same type as `list_t`.
*  <b>`global_norm`</b>: A 0-D (scalar) `Tensor` representing the global norm.

##### Raises:


*  <b>`TypeError`</b>: If `t_list` is not a sequence.

