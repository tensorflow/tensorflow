# TensorFlow contrib losses.

## losses

Loss operations, typically with the following signatures. `predicted` and
`target` generally have the same dimensions, and dim 0 is assumed to be batch.

`squared(predicted, target, name=None) : Tensor`

Other examples of foo are `absolute`, `logistic`, and `softmax`.

Any parameter named `logit` should be the raw model outputs, not a normalized
probablility distribution (i.e., `[0.0, 1.0]`). `target` for losses taking
`logit` _should_ be a normalized probability distribution.
