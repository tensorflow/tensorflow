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

*   @{tf.contrib.distributions.bijectors.Affine}
*   @{tf.contrib.distributions.bijectors.AffineLinearOperator}
*   @{tf.contrib.distributions.bijectors.Bijector}
*   @{tf.contrib.distributions.bijectors.Chain}
*   @{tf.contrib.distributions.bijectors.CholeskyOuterProduct}
*   @{tf.contrib.distributions.bijectors.Exp}
*   @{tf.contrib.distributions.bijectors.Identity}
*   @{tf.contrib.distributions.bijectors.Inline}
*   @{tf.contrib.distributions.bijectors.Invert}
*   @{tf.contrib.distributions.bijectors.PowerTransform}
*   @{tf.contrib.distributions.bijectors.SigmoidCentered}
*   @{tf.contrib.distributions.bijectors.SoftmaxCentered}
*   @{tf.contrib.distributions.bijectors.Softplus}
