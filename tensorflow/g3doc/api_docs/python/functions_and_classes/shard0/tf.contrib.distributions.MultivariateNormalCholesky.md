The multivariate normal distribution on `R^k`.

This distribution is defined by a 1-D mean `mu` and a Cholesky factor `chol`.
Providing the Cholesky factor allows for `O(k^2)` pdf evaluation and sampling,
and requires `O(k^2)` storage.

#### Mathematical details

The Cholesky factor `chol` defines the covariance matrix: `C = chol chol^T`.

The PDF of this distribution is then:

```
f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
```

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and a covariance matrix of shape `k x k`.

Extra leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with diagonal covariance.
# Note, this would be more efficient with MultivariateNormalDiag.
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

#### `tf.contrib.distributions.MultivariateNormalCholesky.__init__(mu, chol, validate_args=True, allow_nan_stats=False, name='MultivariateNormalCholesky')` {#MultivariateNormalCholesky.__init__}

Multivariate Normal distributions on `R^k`.

User must provide means `mu` and `chol` which holds the (batch) Cholesky
factors, such that the covariance of each batch member is `chol chol^T`.

##### Args:


*  <b>`mu`</b>: `(N+1)-D` floating point tensor with shape `[N1,...,Nb, k]`,
    `b >= 0`.
*  <b>`chol`</b>: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
    `[N1,...,Nb, k, k]`.  The upper triangular part is ignored (treated as
    though it is zero), and the diagonal must be positive.
*  <b>`validate_args`</b>: Whether to validate input with asserts.  If `validate_args`
    is `False`, and the inputs are invalid, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: `Boolean`, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: If `mu` and `chol` are different dtypes.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.allow_nan_stats` {#MultivariateNormalCholesky.allow_nan_stats}

`Boolean` describing behavior when stats are undefined.


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

#### `tf.contrib.distributions.MultivariateNormalCholesky.from_params(cls, make_safe=True, **kwargs)` {#MultivariateNormalCholesky.from_params}

Given (unconstrained) parameters, return an instantiated distribution.

Subclasses should implement a static method `_safe_transforms` that returns
a dict of parameter transforms, which will be used if `make_safe = True`.

Example usage:

```
# Let's say we want a sample of size (batch_size, 10)
shapes = MultiVariateNormalDiag.param_shapes([batch_size, 10])

# shapes has a Tensor shape for mu and sigma
# shapes == {
#   'mu': tf.constant([batch_size, 10]),
#   'sigma': tf.constant([batch_size, 10]),
# }

# Here we parameterize mu and sigma with the output of a linear
# layer. Note that sigma is unconstrained.
params = {}
for name, shape in shapes.items():
  params[name] = linear(x, shape[1])

# Note that you can forward other kwargs to the `Distribution`, like
# `allow_nan_stats` or `name`.
mvn = MultiVariateNormalDiag.from_params(**params, allow_nan_stats=True)
```

Distribution parameters may have constraints (e.g. `sigma` must be positive
for a `Normal` distribution) and the `from_params` method will apply default
parameter transforms. If a user wants to use their own transform, they can
apply it externally and set `make_safe=False`.

##### Args:


*  <b>`make_safe`</b>: Whether the `params` should be constrained. If True,
    `from_params` will apply default parameter transforms. If False, no
    parameter transforms will be applied.
*  <b>`**kwargs`</b>: dict of parameters for the distribution.

##### Returns:

  A distribution parameterized by possibly transformed parameters in
  `kwargs`.

##### Raises:


*  <b>`TypeError`</b>: if `make_safe` is `True` but `_safe_transforms` is not
    implemented directly for `cls`.


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

#### `tf.contrib.distributions.MultivariateNormalCholesky.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#MultivariateNormalCholesky.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.param_static_shapes(cls, sample_shape)` {#MultivariateNormalCholesky.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.pdf(value, name='pdf')` {#MultivariateNormalCholesky.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.pmf(value, name='pmf')` {#MultivariateNormalCholesky.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.prob(x, name='prob')` {#MultivariateNormalCholesky.prob}

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

#### `tf.contrib.distributions.MultivariateNormalCholesky.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalCholesky.sample}

Generate samples of the specified shape for each batched distribution.

Note that a call to `sample()` without arguments will generate a single
sample per batched distribution.

##### Args:


*  <b>`sample_shape`</b>: Rank 1 `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of dtype `self.dtype` and shape
      `sample_shape + self.batch_shape + self.event_shape`.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.sample_n(n, seed=None, name='sample_n')` {#MultivariateNormalCholesky.sample_n}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar` `Tensor` of type `int32` or `int64`, the number of
    observations to sample.
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

#### `tf.contrib.distributions.MultivariateNormalCholesky.validate_args` {#MultivariateNormalCholesky.validate_args}

`Boolean` describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.MultivariateNormalCholesky.variance(name='variance')` {#MultivariateNormalCholesky.variance}

Variance of each batch member.


