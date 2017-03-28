# Random variable transformations (contrib)
[TOC]

Bijector Ops.

An API for invertible, differentiable transformations of random variables.

## Background

Differentiable, bijective transformations of continuous random variables alter
the calculations made in the cumulative/probability distribution functions and
sample function.  This module provides a standard interface for making these
manipulations.

For more details and examples, see the `Bijector` docstring.

To apply a `Bijector`, use `distributions.TransformedDistribution`.

## Bijectors

*   @{tf.contrib.distributions.bijector.Affine}
*   @{tf.contrib.distributions.bijector.AffineLinearOperator}
*   @{tf.contrib.distributions.bijector.Bijector}
*   @{tf.contrib.distributions.bijector.Chain}
*   @{tf.contrib.distributions.bijector.CholeskyOuterProduct}
*   @{tf.contrib.distributions.bijector.Exp}
*   @{tf.contrib.distributions.bijector.Identity}
*   @{tf.contrib.distributions.bijector.Inline}
*   @{tf.contrib.distributions.bijector.Invert}
*   @{tf.contrib.distributions.bijector.PowerTransform}
*   @{tf.contrib.distributions.bijector.SigmoidCentered}
*   @{tf.contrib.distributions.bijector.SoftmaxCentered}
*   @{tf.contrib.distributions.bijector.Softplus}
