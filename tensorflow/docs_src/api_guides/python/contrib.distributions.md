# Statistical Distributions (contrib)
[TOC]

Classes representing statistical distributions and ops for working with them.

## Classes for statistical distributions

Classes that represent batches of statistical distributions.  Each class is
initialized with parameters that define the distributions.

## Base classes

*   @{tf.contrib.distributions.ReparameterizationType}
*   @{tf.contrib.distributions.Distribution}

## Univariate (scalar) distributions

*   @{tf.contrib.distributions.Binomial}
*   @{tf.contrib.distributions.Bernoulli}
*   @{tf.contrib.distributions.BernoulliWithSigmoidProbs}
*   @{tf.contrib.distributions.Beta}
*   @{tf.contrib.distributions.Categorical}
*   @{tf.contrib.distributions.Chi2}
*   @{tf.contrib.distributions.Chi2WithAbsDf}
*   @{tf.contrib.distributions.Exponential}
*   @{tf.contrib.distributions.Gamma}
*   @{tf.contrib.distributions.InverseGamma}
*   @{tf.contrib.distributions.Laplace}
*   @{tf.contrib.distributions.LaplaceWithSoftplusScale}
*   @{tf.contrib.distributions.Normal}
*   @{tf.contrib.distributions.NormalWithSoftplusScale}
*   @{tf.contrib.distributions.Poisson}
*   @{tf.contrib.distributions.StudentT}
*   @{tf.contrib.distributions.StudentTWithAbsDfSoftplusScale}
*   @{tf.contrib.distributions.Uniform}

## Multivariate distributions

### Multivariate normal

*   @{tf.contrib.distributions.MultivariateNormalDiag}
*   @{tf.contrib.distributions.MultivariateNormalTriL}
*   @{tf.contrib.distributions.MultivariateNormalDiagPlusLowRank}
*   @{tf.contrib.distributions.MultivariateNormalDiagWithSoftplusScale}

### Other multivariate distributions

*   @{tf.contrib.distributions.Dirichlet}
*   @{tf.contrib.distributions.DirichletMultinomial}
*   @{tf.contrib.distributions.Multinomial}
*   @{tf.contrib.distributions.WishartCholesky}
*   @{tf.contrib.distributions.WishartFull}

### Multivariate Utilities

*   @{tf.contrib.distributions.matrix_diag_transform}

## Transformed distributions

*   @{tf.contrib.distributions.TransformedDistribution}
*   @{tf.contrib.distributions.QuantizedDistribution}

## Mixture Models

*   @{tf.contrib.distributions.Mixture}

## Posterior inference with conjugate priors

Functions that transform conjugate prior/likelihood pairs to distributions
representing the posterior or posterior predictive.

## Normal likelihood with conjugate prior

*   @{tf.contrib.distributions.normal_conjugates_known_scale_posterior}
*   @{tf.contrib.distributions.normal_conjugates_known_scale_predictive}

## Kullback-Leibler Divergence

*   @{tf.contrib.distributions.kl}
*   @{tf.contrib.distributions.RegisterKL}

## Utilities

*   @{tf.contrib.distributions.softplus_inverse}
