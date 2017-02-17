# TensorFlow contrib losses.

## losses

Loss operations for use in training models, typically with signature like the
following:

`sum_of_squares(predictions, labels, weight, scope) : Tensor`

All loss functions take a pair of tensors, `predictions` and ground truth
`labels`. It is assumed that the shape of both these tensors is of the form
`[batch_size, d1, ... dN]` where `batch_size` is the number
of samples in the batch and `d1` ... `dN` are the remaining dimensions.

THe `weight` parameter can be used to adjust the relative weight samples within
the batch. The result of each loss is a scalar average of all sample losses with
non-zero weights.

Any parameter named `logit` should be the raw model outputs, not a normalized
probablility distribution (i.e., `[0.0, 1.0]`). `target` for losses taking
`logit` _should_ be a normalized probability distribution.
