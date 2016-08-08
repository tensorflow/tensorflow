The multivariate normal distribution on `R^k`.

Every batch member of this distribution is defined by a mean and a lightweight
covariance matrix `C`.

#### Mathematical details

The PDF of this distribution in terms of the mean `mu` and covariance `C` is:

```
f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
```

For every batch member, this distribution represents `k` random variables
`(X_1,...,X_k)`, with mean `E[X_i] = mu[i]`, and covariance matrix
`C_{ij} := E[(X_i - mu[i])(X_j - mu[j])]`

The user initializes this class by providing the mean `mu`, and a lightweight
definition of `C`:

```
C = SS^T = SS = (M + V D V^T) (M + V D V^T)
M is diagonal (k x k)
V = is shape (k x r), typically r << k
D = is diagonal (r x r), optional (defaults to identity).
```

This allows for `O(kr + r^3)` pdf evaluation and determinant, and `O(kr)`
sampling and storage (per batch member).

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and square root of the covariance `S = M + V D V^T`.  Extra
leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with covariance square root
# S = M + V D V^T, where V D V^T is a matrix-rank 2 update.
mu = [1, 2, 3.]
diag_large = [1.1, 2.2, 3.3]
v = ... # shape 3 x 2
diag_small = [4., 5.]
dist = tf.contrib.distributions.MultivariateNormalDiagPlusVDVT(
    mu, diag_large, v, diag_small=diag_small)

# Evaluate this on an observation in R^3, returning a scalar.
dist.pdf([-1, 0, 1])

# Initialize a batch of two 3-variate Gaussians.  This time, don't provide
# diag_small.  This means S = M + V V^T.
mu = [[1, 2, 3], [11, 22, 33]]  # shape 2 x 3
diag_large = ... # shape 2 x 3
v = ... # shape 2 x 3 x 1, a matrix-rank 1 update.
dist = tf.contrib.distributions.MultivariateNormalDiagPlusVDVT(
    mu, diag_large, v)

# Evaluate this on a two observations, each in R^3, returning a length two
# tensor.
x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
dist.pdf(x)
```
- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.__init__(mu, diag_large, v, diag_small=None, validate_args=True, allow_nan_stats=False, name='MultivariateNormalDiagPlusVDVT')` {#MultivariateNormalDiagPlusVDVT.__init__}

Multivariate Normal distributions on `R^k`.

For every batch member, this distribution represents `k` random variables
`(X_1,...,X_k)`, with mean `E[X_i] = mu[i]`, and covariance matrix
`C_{ij} := E[(X_i - mu[i])(X_j - mu[j])]`

The user initializes this class by providing the mean `mu`, and a
lightweight definition of `C`:

```
C = SS^T = SS = (M + V D V^T) (M + V D V^T)
M is diagonal (k x k)
V = is shape (k x r), typically r << k
D = is diagonal (r x r), optional (defaults to identity).
```

##### Args:


*  <b>`mu`</b>: Rank `n + 1` floating point tensor with shape `[N1,...,Nn, k]`,
    `n >= 0`.  The means.
*  <b>`diag_large`</b>: Optional rank `n + 1` floating point tensor, shape
    `[N1,...,Nn, k]` `n >= 0`.  Defines the diagonal matrix `M`.
*  <b>`v`</b>: Rank `n + 1` floating point tensor, shape `[N1,...,Nn, k, r]`
    `n >= 0`.  Defines the matrix `V`.
*  <b>`diag_small`</b>: Rank `n + 1` floating point tensor, shape
    `[N1,...,Nn, k]` `n >= 0`.  Defines the diagonal matrix `D`.  Default
    is `None`, which means `D` will be the identity matrix.
*  <b>`validate_args`</b>: Whether to validate input with asserts.  If `validate_args`
    is `False`,
    and the inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: `Boolean`, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.allow_nan_stats` {#MultivariateNormalDiagPlusVDVT.allow_nan_stats}

`Boolean` describing behavior when stats are undefined.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.batch_shape(name='batch_shape')` {#MultivariateNormalDiagPlusVDVT.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.cdf(value, name='cdf')` {#MultivariateNormalDiagPlusVDVT.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.dtype` {#MultivariateNormalDiagPlusVDVT.dtype}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.entropy(name='entropy')` {#MultivariateNormalDiagPlusVDVT.entropy}

The entropies of these Multivariate Normals.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropies.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.event_shape(name='event_shape')` {#MultivariateNormalDiagPlusVDVT.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.get_batch_shape()` {#MultivariateNormalDiagPlusVDVT.get_batch_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.get_event_shape()` {#MultivariateNormalDiagPlusVDVT.get_event_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.is_continuous` {#MultivariateNormalDiagPlusVDVT.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.is_reparameterized` {#MultivariateNormalDiagPlusVDVT.is_reparameterized}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.log_cdf(value, name='log_cdf')` {#MultivariateNormalDiagPlusVDVT.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.log_pdf(value, name='log_pdf')` {#MultivariateNormalDiagPlusVDVT.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.log_pmf(value, name='log_pmf')` {#MultivariateNormalDiagPlusVDVT.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.log_prob(x, name='log_prob')` {#MultivariateNormalDiagPlusVDVT.log_prob}

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

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.log_sigma_det(name='log_sigma_det')` {#MultivariateNormalDiagPlusVDVT.log_sigma_det}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.mean(name='mean')` {#MultivariateNormalDiagPlusVDVT.mean}

Mean of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.mode(name='mode')` {#MultivariateNormalDiagPlusVDVT.mode}

Mode of each batch member.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.mu` {#MultivariateNormalDiagPlusVDVT.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.name` {#MultivariateNormalDiagPlusVDVT.name}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.pdf(value, name='pdf')` {#MultivariateNormalDiagPlusVDVT.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.pmf(value, name='pmf')` {#MultivariateNormalDiagPlusVDVT.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.prob(x, name='prob')` {#MultivariateNormalDiagPlusVDVT.prob}

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

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalDiagPlusVDVT.sample}

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

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.sample_n(n, seed=None, name='sample_n')` {#MultivariateNormalDiagPlusVDVT.sample_n}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.sigma` {#MultivariateNormalDiagPlusVDVT.sigma}

Dense (batch) covariance matrix, if available.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.sigma_det(name='sigma_det')` {#MultivariateNormalDiagPlusVDVT.sigma_det}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.std(name='std')` {#MultivariateNormalDiagPlusVDVT.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.validate_args` {#MultivariateNormalDiagPlusVDVT.validate_args}

`Boolean` describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagPlusVDVT.variance(name='variance')` {#MultivariateNormalDiagPlusVDVT.variance}

Variance of each batch member.


