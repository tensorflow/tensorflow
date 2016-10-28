### `tf.einsum(axes, *inputs)` {#einsum}

A generalized contraction between tensors of arbitrary dimension.

Like `numpy.einsum`, but does not support:
* Ellipses (subscripts like `ij...,jk...->ik...`)
* Subscripts where an axis appears more than once for a single input (e.g. `ijj,jk->ik`).

##### Args:


*  <b>`axes`</b>: a `str` describing the contraction, in the same format as `numpy.einsum`.
*  <b>`inputs`</b>: the inputs to contract (each one a `Tensor`), whose shapes should be consistent with `axes`.

##### Returns:

  The contracted `Tensor`, with shape determined by `axes`.

##### Raises:


*  <b>`ValueError`</b>: If the format of `axes` is incorrect,
              or the number of inputs implied by `axes` does not match `len(inputs)`,
              or an axis appears in the output subscripts but not in any of the inputs,
              or the number of dimensions of an input differs from the number of indices in its subscript,
              or the input shapes are inconsistent along a particular axis.

