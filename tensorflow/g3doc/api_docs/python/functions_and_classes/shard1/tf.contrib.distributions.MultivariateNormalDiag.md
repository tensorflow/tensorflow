The multivariate normal distribution on `R^k`.

This distribution is defined by a 1-D mean `mu` and a 1-D diagonal
`diag_stdev`, representing the standard deviations.  This distribution
assumes the random variables, `(X_1,...,X_k)` are independent, thus no
non-diagonal terms of the covariance matrix are needed.

This allows for `O(k)` pdf evaluation, sampling, and storage.

#### Mathematical details

The PDF of this distribution is defined in terms of the diagonal covariance
determined by `diag_stdev`: `C_{ii} = diag_stdev[i]**2`.

```
f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
```

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and the square roots of the (independent) random variables.

Extra leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with diagonal standard deviation.
mu = [1, 2, 3.]
diag_stdev = [4, 5, 6.]
dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_stdev)

# Evaluate this on an observation in R^3, returning a scalar.
dist.pdf([-1, 0, 1])

# Initialize a batch of two 3-variate Gaussians.
mu = [[1, 2, 3], [11, 22, 33]]  # shape 2 x 3
diag_stdev = ...  # shape 2 x 3, positive.
dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_stdev)

# Evaluate this on a two observations, each in R^3, returning a length two
# tensor.
x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
dist.pdf(x)
```
- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.__init__(mu, diag_stdev, validate_args=True, allow_nan_stats=False, name='MultivariateNormalDiag')` {#MultivariateNormalDiag.__init__}

Multivariate Normal distributions on `R^k`.

User must provide means `mu` and standard deviations `diag_stdev`.
Each batch member represents a random vector `(X_1,...,X_k)` of independent
random normals.
The mean of `X_i` is `mu[i]`, and the standard deviation is `diag_stdev[i]`.

##### Args:


*  <b>`mu`</b>: Rank `N + 1` floating point tensor with shape `[N1,...,Nb, k]`,
    `b >= 0`.
*  <b>`diag_stdev`</b>: Rank `N + 1` `Tensor` with same `dtype` and shape as `mu`,
    representing the standard deviations.  Must be positive.
*  <b>`validate_args`</b>: Whether to validate input with asserts.  If `validate_args`
    is `False`,
    and the inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: `Boolean`, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: If `mu` and `diag_stdev` are different dtypes.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.allow_nan_stats` {#MultivariateNormalDiag.allow_nan_stats}

`Boolean` describing behavior when stats are undefined.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.batch_shape(name='batch_shape')` {#MultivariateNormalDiag.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.cdf(value, name='cdf')` {#MultivariateNormalDiag.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.dtype` {#MultivariateNormalDiag.dtype}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.entropy(name='entropy')` {#MultivariateNormalDiag.entropy}

The entropies of these Multivariate Normals.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropies.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.event_shape(name='event_shape')` {#MultivariateNormalDiag.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.get_batch_shape()` {#MultivariateNormalDiag.get_batch_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.get_event_shape()` {#MultivariateNormalDiag.get_event_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.is_continuous` {#MultivariateNormalDiag.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.is_reparameterized` {#MultivariateNormalDiag.is_reparameterized}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.log_cdf(value, name='log_cdf')` {#MultivariateNormalDiag.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.log_pdf(value, name='log_pdf')` {#MultivariateNormalDiag.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.log_pmf(value, name='log_pmf')` {#MultivariateNormalDiag.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.log_prob(x, name='log_prob')` {#MultivariateNormalDiag.log_prob}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.log_sigma_det(name='log_sigma_det')` {#MultivariateNormalDiag.log_sigma_det}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.mean(name='mean')` {#MultivariateNormalDiag.mean}

Mean of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.mode(name='mode')` {#MultivariateNormalDiag.mode}

Mode of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.mu` {#MultivariateNormalDiag.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.name` {#MultivariateNormalDiag.name}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.pdf(value, name='pdf')` {#MultivariateNormalDiag.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.pmf(value, name='pmf')` {#MultivariateNormalDiag.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.prob(x, name='prob')` {#MultivariateNormalDiag.prob}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalDiag.sample}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.sample_n(n, seed=None, name='sample_n')` {#MultivariateNormalDiag.sample_n}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.sigma` {#MultivariateNormalDiag.sigma}

Dense (batch) covariance matrix, if available.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.sigma_det(name='sigma_det')` {#MultivariateNormalDiag.sigma_det}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.std(name='std')` {#MultivariateNormalDiag.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.validate_args` {#MultivariateNormalDiag.validate_args}

`Boolean` describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.variance(name='variance')` {#MultivariateNormalDiag.variance}

Variance of each batch member.


