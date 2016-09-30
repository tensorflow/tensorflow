### `tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)` {#batch_normalization}

Batch normalization.

As described in http://arxiv.org/abs/1502.03167.
Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
`scale` \\\\(\gamma\\\\) to it, as well as an `offset` \\\\(\\beta\\\\):

\\\\(\\frac{\gamma(x-\mu)}{\sigma}+\\beta\\\\)

`mean`, `variance`, `offset` and `scale` are all expected to be of one of two
shapes:

  * In all generality, they can have the same number of dimensions as the
    input `x`, with identical sizes as `x` for the dimensions that are not
    normalized over (the 'depth' dimension(s)), and dimension 1 for the
    others which are being normalized over.
    `mean` and `variance` in this case would typically be the outputs of
    `tf.nn.moments(..., keep_dims=True)` during training, or running averages
    thereof during inference.
  * In the common case where the 'depth' dimension is the last dimension in
    the input tensor `x`, they may be one dimensional tensors of the same
    size as the 'depth' dimension.
    This is the case for example for the common `[batch, depth]` layout of
    fully-connected layers, and `[batch, height, width, depth]` for
    convolutions.
    `mean` and `variance` in this case would typically be the outputs of
    `tf.nn.moments(..., keep_dims=False)` during training, or running averages
    thereof during inference.

##### Args:


*  <b>`x`</b>: Input `Tensor` of arbitrary dimensionality.
*  <b>`mean`</b>: A mean `Tensor`.
*  <b>`variance`</b>: A variance `Tensor`.
*  <b>`offset`</b>: An offset `Tensor`, often denoted \\\\(\\beta\\\\) in equations, or
    None. If present, will be added to the normalized tensor.
*  <b>`scale`</b>: A scale `Tensor`, often denoted \\\\(\gamma\\\\) in equations, or
    `None`. If present, the scale is applied to the normalized tensor.
*  <b>`variance_epsilon`</b>: A small float number to avoid dividing by 0.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  the normalized, scaled, offset tensor.

