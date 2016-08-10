The multivariate normal distribution on `R^k`.

This distribution is defined by a 1-D mean `mu` and covariance matrix `sigma`.
Evaluation of the pdf, determinant, and sampling are all `O(k^3)` operations.

#### Mathematical details

With `C = sigma`, the PDF of this distribution is:

```
f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
```

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and a covariance matrix of shape `k x k`.

Extra leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with diagonal covariance.
mu = [1, 2, 3.]
sigma = [[1, 0, 0], [0, 3, 0], [0, 0, 2.]]
dist = tf.contrib.distributions.MultivariateNormalFull(mu, chol)

# Evaluate this on an observation in R^3, returning a scalar.
dist.pdf([-1, 0, 1])

# Initialize a batch of two 3-variate Gaussians.
mu = [[1, 2, 3], [11, 22, 33.]]
sigma = ...  # shape 2 x 3 x 3, positive definite.
dist = tf.contrib.distributions.MultivariateNormalFull(mu, sigma)

# Evaluate this on a two observations, each in R^3, returning a length two
# tensor.
x = [[-1, 0, 1], [-11, 0, 11.]]  # Shape 2 x 3.
dist.pdf(x)
```
- - -

#### `tf.contrib.distributions.MultivariateNormalFull.__init__(mu, sigma, validate_args=True, allow_nan_stats=False, name='MultivariateNormalFull')` {#MultivariateNormalFull.__init__}

Multivariate Normal distributions on `R^k`.

User must provide means `mu` and `sigma`, the mean and covariance.

##### Args:


*  <b>`mu`</b>: `(N+1)-D` floating point tensor with shape `[N1,...,Nb, k]`,
    `b >= 0`.
*  <b>`sigma`</b>: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
    `[N1,...,Nb, k, k]`.  Each batch member must be positive definite.
*  <b>`validate_args`</b>: Whether to validate input with asserts.  If `validate_args`
    is `False`, and the inputs are invalid, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: `Boolean`, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: If `mu` and `sigma` are different dtypes.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.allow_nan_stats` {#MultivariateNormalFull.allow_nan_stats}

`Boolean` describing behavior when stats are undefined.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.batch_shape(name='batch_shape')` {#MultivariateNormalFull.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.cdf(value, name='cdf')` {#MultivariateNormalFull.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.dtype` {#MultivariateNormalFull.dtype}




- - -

#### `tf.contrib.distributions.MultivariateNormalFull.entropy(name='entropy')` {#MultivariateNormalFull.entropy}

The entropies of these Multivariate Normals.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropies.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.event_shape(name='event_shape')` {#MultivariateNormalFull.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.get_batch_shape()` {#MultivariateNormalFull.get_batch_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.get_event_shape()` {#MultivariateNormalFull.get_event_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.is_continuous` {#MultivariateNormalFull.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalFull.is_reparameterized` {#MultivariateNormalFull.is_reparameterized}




- - -

#### `tf.contrib.distributions.MultivariateNormalFull.log_cdf(value, name='log_cdf')` {#MultivariateNormalFull.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.log_pdf(value, name='log_pdf')` {#MultivariateNormalFull.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.log_pmf(value, name='log_pmf')` {#MultivariateNormalFull.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.log_prob(x, name='log_prob')` {#MultivariateNormalFull.log_prob}

Log prob of observations `x` given these Multivariate Normals.

`x` is a batch vector with compatible shape if `x` is a `Tensor` whose
shape can be broadcast up to either:

````
self.batch_shape + self.event_shape
OR
[M1,...,Mm] + self.batch_shape + self.event_shape
```

##### Args:


*  <b>`x`</b>: Compatible batch vector with same `dtype` as this distribution.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.log_sigma_det(name='log_sigma_det')` {#MultivariateNormalFull.log_sigma_det}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.mean(name='mean')` {#MultivariateNormalFull.mean}

Mean of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.mode(name='mode')` {#MultivariateNormalFull.mode}

Mode of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.mu` {#MultivariateNormalFull.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormalFull.name` {#MultivariateNormalFull.name}




- - -

#### `tf.contrib.distributions.MultivariateNormalFull.pdf(value, name='pdf')` {#MultivariateNormalFull.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.pmf(value, name='pmf')` {#MultivariateNormalFull.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.prob(x, name='prob')` {#MultivariateNormalFull.prob}

The PDF of observations `x` under these Multivariate Normals.

`x` is a batch vector with compatible shape if `x` is a `Tensor` whose
shape can be broadcast up to either:

````
self.batch_shape + self.event_shape
OR
[M1,...,Mm] + self.batch_shape + self.event_shape
```

##### Args:


*  <b>`x`</b>: Compatible batch vector with same `dtype` as this distribution.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the prob values of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalFull.sample}

Generate samples of the specified shape for each batched distribution.

Note that a call to `sample()` without arguments will generate a single
sample per batched distribution.

##### Args:


*  <b>`sample_shape`</b>: `int32` `Tensor` or tuple or list. Shape of the generated
    samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of dtype `self.dtype` and shape
      `sample_shape + self.batch_shape + self.event_shape`.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.sample_n(n, seed=None, name='sample_n')` {#MultivariateNormalFull.sample_n}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.sigma` {#MultivariateNormalFull.sigma}

Dense (batch) covariance matrix, if available.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.sigma_det(name='sigma_det')` {#MultivariateNormalFull.sigma_det}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.std(name='std')` {#MultivariateNormalFull.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.validate_args` {#MultivariateNormalFull.validate_args}

`Boolean` describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.MultivariateNormalFull.variance(name='variance')` {#MultivariateNormalFull.variance}

Variance of each batch member.


