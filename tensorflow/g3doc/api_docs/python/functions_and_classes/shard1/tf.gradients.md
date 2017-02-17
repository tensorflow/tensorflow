### `tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)` {#gradients}

Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`.

`ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
is a list of `Tensor`, holding the gradients received by the
`ys`. The list must be the same length as `ys`.

`gradients()` adds ops to the graph to output the partial
derivatives of `ys` with respect to `xs`.  It returns a list of
`Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
for y in `ys`.

`grad_ys` is a list of tensors of the same length as `ys` that holds
the initial gradients for each y in `ys`.  When `grad_ys` is None,
we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
user can provide their own initial `grad_ys` to compute the
derivatives using a different initial gradient for each y (e.g., if
one wanted to weight the gradient differently for each value in
each y).

##### Args:


*  <b>`ys`</b>: A `Tensor` or list of tensors to be differentiated.
*  <b>`xs`</b>: A `Tensor` or list of tensors to be used for differentiation.
*  <b>`grad_ys`</b>: Optional. A `Tensor` or list of tensors the same size as
    `ys` and holding the gradients computed for each y in `ys`.
*  <b>`name`</b>: Optional name to use for grouping all the gradient ops together.
    defaults to 'gradients'.
*  <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with
    the corresponding op.
*  <b>`gate_gradients`</b>: If True, add a tuple around the gradients returned
    for an operations.  This avoids some race conditions.
*  <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Accepted values are constants defined in the class `AggregationMethod`.

##### Returns:

  A list of `sum(dy/dx)` for each x in `xs`.

##### Raises:


*  <b>`LookupError`</b>: if one of the operations between `x` and `y` does not
    have a registered gradient function.
*  <b>`ValueError`</b>: if the arguments are invalid.

