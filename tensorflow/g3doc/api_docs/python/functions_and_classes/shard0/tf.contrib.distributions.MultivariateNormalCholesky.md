The multivariate normal distribution on `R^k`.

This distribution is defined by a 1-D mean `mu` and a Cholesky factor `chol`.
Providing the Cholesky factor allows for `O(k^2)` pdf evaluation and sampling,
and requires `O(k^2)` storage.

#### Mathematical details

The PDF of this distribution is:

```
f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
```

where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and a covariance matrix of shape `k x k`.

Extra leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with diagonal covariance.
mu = [1, 2, 3.]
chol = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
dist = tf.contrib.distributions.MultivariateNormalCholesky(mu, chol)

# Evaluate this on an observation in R^3, returning a scalar.
dist.pdf([-1, 0, 1])

# Initialize a batch of two 3-variate Gaussians.
mu = [[1, 2, 3], [11, 22, 33]]
chol = ...  # shape 2 x 3 x 3, lower triangular, positive diagonal.
dist = tf.contrib.distributions.MultivariateNormalCholesky(mu, chol)

# Evaluate this on a two observations, each in R^3, returning a length two
# tensor.
x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
dist.pdf(x)
```

Trainable (batch) Choesky matrices can be created with
`tf.contrib.distributions.batch_matrix_diag_transform()`
- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.__init__(mu, chol, strict=True, strict_statistics=True, name='MultivariateNormalCholesky')` {#MultivariateNormalCholesky.__init__}

Multivariate Normal distributions on `R^k`.

User must provide means `mu` and `chol` which holds the (batch) Cholesky
factors `S`, such that the covariance of each batch member is `S S^*`.

##### Args:


*  <b>`mu`</b>: `(N+1)-D`  `float` or `double` tensor with shape `[N1,...,Nb, k]`,
    `b >= 0`.
*  <b>`chol`</b>: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
    `[N1,...,Nb, k, k]`.
*  <b>`strict`</b>: Whether to validate input with asserts.  If `strict` is `False`,
    and the inputs are invalid, correct behavior is not guaranteed.
*  <b>`strict_statistics`</b>: Boolean, default True.  If True, raise an exception if
    a statistic (e.g. mean/mode/etc...) is undefined for any batch member.
    If False, batch members with valid parameters leading to undefined
    statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: If `mu` and `chol` are different dtypes.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.batch_shape(name='batch_shape')` {#MultivariateNormalCholesky.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.cdf(value, name='cdf')` {#MultivariateNormalCholesky.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.dtype` {#MultivariateNormalCholesky.dtype}




- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.entropy(name='entropy')` {#MultivariateNormalCholesky.entropy}

The entropies of these Multivariate Normals.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropies.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.event_shape(name='event_shape')` {#MultivariateNormalCholesky.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.get_batch_shape()` {#MultivariateNormalCholesky.get_batch_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.get_event_shape()` {#MultivariateNormalCholesky.get_event_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.is_continuous` {#MultivariateNormalCholesky.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.is_reparameterized` {#MultivariateNormalCholesky.is_reparameterized}




- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.log_cdf(value, name='log_cdf')` {#MultivariateNormalCholesky.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.log_pdf(value, name='log_pdf')` {#MultivariateNormalCholesky.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.log_pmf(value, name='log_pmf')` {#MultivariateNormalCholesky.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.log_prob(x, name='log_prob')` {#MultivariateNormalCholesky.log_prob}

Log prob of observations `x` given these Multivariate Normals.

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.log_sigma_det(name='log_sigma_det')` {#MultivariateNormalCholesky.log_sigma_det}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.mean(name='mean')` {#MultivariateNormalCholesky.mean}

Mean of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.mode(name='mode')` {#MultivariateNormalCholesky.mode}

Mode of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.mu` {#MultivariateNormalCholesky.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.name` {#MultivariateNormalCholesky.name}




- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.pdf(value, name='pdf')` {#MultivariateNormalCholesky.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.pmf(value, name='pmf')` {#MultivariateNormalCholesky.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.prob(x, name='prob')` {#MultivariateNormalCholesky.prob}

The PDF of observations `x` under these Multivariate Normals.

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the prob values of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.sample(n, seed=None, name='sample')` {#MultivariateNormalCholesky.sample}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.sigma` {#MultivariateNormalCholesky.sigma}

Dense (batch) covariance matrix, if available.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.sigma_det(name='sigma_det')` {#MultivariateNormalCholesky.sigma_det}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.std(name='std')` {#MultivariateNormalCholesky.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.strict` {#MultivariateNormalCholesky.strict}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.strict_statistics` {#MultivariateNormalCholesky.strict_statistics}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.variance(name='variance')` {#MultivariateNormalCholesky.variance}

Variance of each batch member.


