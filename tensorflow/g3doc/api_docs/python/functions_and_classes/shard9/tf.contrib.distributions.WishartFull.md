The matrix Wishart distribution on positive definite matrices.

This distribution is defined by a scalar degrees of freedom `df` and a
symmetric, positive definite scale matrix.

Evaluation of the pdf, determinant, and sampling are all `O(k^3)` operations
where `(k, k)` is the event space shape.

#### Mathematical details.

The PDF of this distribution is,

```
f(X) = det(X)^(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / B(scale, df)
```

where `df >= k` denotes the degrees of freedom, `scale` is a symmetric, pd,
`k x k` matrix, and the normalizing constant `B(scale, df)` is given by:

```
B(scale, df) = 2^(0.5 df k) |det(scale)|^(0.5 df) Gamma_k(0.5 df)
```

where `Gamma_k` is the multivariate Gamma function.

#### Examples

```python
# Initialize a single 3x3 Wishart with Full factored scale matrix and 5
# degrees-of-freedom.(*)
df = 5
scale = ...  # Shape is [3, 3]; positive definite.
dist = tf.contrib.distributions.WishartFull(df=df, scale=scale)

# Evaluate this on an observation in R^3, returning a scalar.
x = ... # A 3x3 positive definite matrix.
dist.pdf(x)  # Shape is [], a scalar.

# Evaluate this on a two observations, each in R^{3x3}, returning a length two
# Tensor.
x = [x0, x1]  # Shape is [2, 3, 3].
dist.pdf(x)  # Shape is [2].

# Initialize two 3x3 Wisharts with Full factored scale matrices.
df = [5, 4]
scale = ...  # Shape is [2, 3, 3].
dist = tf.contrib.distributions.WishartFull(df=df, scale=scale)

# Evaluate this on four observations.
x = [[x0, x1], [x2, x3]]  # Shape is [2, 2, 3, 3]; xi is positive definite.
dist.pdf(x)  # Shape is [2, 2].

# (*) - To efficiently create a trainable covariance matrix, see the example
#   in tf.contrib.distributions.batch_matrix_diag_transform.
```
- - -

#### `tf.contrib.distributions.WishartFull.__init__(df, scale, cholesky_input_output_matrices=False, allow_nan_stats=False, validate_args=True, name='Wishart')` {#WishartFull.__init__}

Construct Wishart distributions.

##### Args:


*  <b>`df`</b>: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
    or equal to dimension of the scale matrix.
*  <b>`scale`</b>: `float` or `double` `Tensor`. The symmetric positive definite
    scale matrix of the distribution.
*  <b>`cholesky_input_output_matrices`</b>: `Boolean`. Any function which whose input
    or output is a matrix assumes the input is Cholesky and returns a
    Cholesky factored matrix. Example`log_pdf` input takes a Cholesky and
    `sample_n` returns a Cholesky when
    `cholesky_input_output_matrices=True`.
*  <b>`allow_nan_stats`</b>: `Boolean`, default `False`. If `False`, raise an
    exception if a statistic (e.g., mean, mode) is undefined for any batch
    member. If True, batch members with valid parameters leading to
    undefined statistics will return `NaN` for this statistic.
*  <b>`validate_args`</b>: Whether to validate input with asserts. If `validate_args`
    is `False`, and the inputs are invalid, correct behavior is not
    guaranteed.
*  <b>`name`</b>: The name scope to give class member ops.


- - -

#### `tf.contrib.distributions.WishartFull.allow_nan_stats` {#WishartFull.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.WishartFull.batch_shape(name='batch_shape')` {#WishartFull.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.WishartFull.cdf(value, name='cdf')` {#WishartFull.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.WishartFull.cholesky_input_output_matrices` {#WishartFull.cholesky_input_output_matrices}

Boolean indicating if `Tensor` input/outputs are Cholesky factorized.


- - -

#### `tf.contrib.distributions.WishartFull.df` {#WishartFull.df}

Wishart distribution degree(s) of freedom.


- - -

#### `tf.contrib.distributions.WishartFull.dimension` {#WishartFull.dimension}

Dimension of underlying vector space. The `p` in `R^(p*p)`.


- - -

#### `tf.contrib.distributions.WishartFull.dtype` {#WishartFull.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.WishartFull.entropy(name='entropy')` {#WishartFull.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.WishartFull.event_shape(name='event_shape')` {#WishartFull.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.


- - -

#### `tf.contrib.distributions.WishartFull.from_params(cls, make_safe=True, **kwargs)` {#WishartFull.from_params}

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

#### `tf.contrib.distributions.WishartFull.get_batch_shape()` {#WishartFull.get_batch_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.WishartFull.get_event_shape()` {#WishartFull.get_event_shape}

`TensorShape` available at graph construction time.


- - -

#### `tf.contrib.distributions.WishartFull.inputs` {#WishartFull.inputs}

Dictionary of inputs provided at initialization.


- - -

#### `tf.contrib.distributions.WishartFull.is_continuous()` {#WishartFull.is_continuous}




- - -

#### `tf.contrib.distributions.WishartFull.is_reparameterized()` {#WishartFull.is_reparameterized}




- - -

#### `tf.contrib.distributions.WishartFull.log_cdf(value, name='log_cdf')` {#WishartFull.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.WishartFull.log_normalizing_constant(name='log_normalizing_constant')` {#WishartFull.log_normalizing_constant}

Computes the log normalizing constant, log(Z).


- - -

#### `tf.contrib.distributions.WishartFull.log_pdf(value, name='log_pdf')` {#WishartFull.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.WishartFull.log_pmf(value, name='log_pmf')` {#WishartFull.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.WishartFull.log_prob(x, name='log_prob')` {#WishartFull.log_prob}

Log of the probability density/mass function.

##### Args:


*  <b>`x`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.WishartFull.mean(name='mean')` {#WishartFull.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.WishartFull.mean_log_det(name='mean_log_det')` {#WishartFull.mean_log_det}

Computes E[log(det(X))] under this Wishart distribution.


- - -

#### `tf.contrib.distributions.WishartFull.mode(name='mode')` {#WishartFull.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.WishartFull.name` {#WishartFull.name}

Name prepended to all ops.


- - -

#### `tf.contrib.distributions.WishartFull.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#WishartFull.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.WishartFull.param_static_shapes(cls, sample_shape)` {#WishartFull.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.WishartFull.pdf(value, name='pdf')` {#WishartFull.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.WishartFull.pmf(value, name='pmf')` {#WishartFull.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.WishartFull.prob(value, name='prob')` {#WishartFull.prob}

Probability density/mass function.


- - -

#### `tf.contrib.distributions.WishartFull.sample(sample_shape=(), seed=None, name='sample')` {#WishartFull.sample}

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

#### `tf.contrib.distributions.WishartFull.sample_n(n, seed=None, name='sample')` {#WishartFull.sample_n}

Generate `n` samples.

Complexity: O(nbk^3)

The sampling procedure is based on the [Bartlett decomposition](
https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition)
and [using a Gamma distribution to generate Chi2 random variates](
https://en.wikipedia.org/wiki/Chi-squared_distribution#Gamma.2C_exponential.2C_and_related_distributions).

##### Args:


*  <b>`n`</b>: Scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer; random number generator seed.
*  <b>`name`</b>: The name of this op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.WishartFull.scale()` {#WishartFull.scale}

Wishart distribution scale matrix.


- - -

#### `tf.contrib.distributions.WishartFull.scale_operator_pd` {#WishartFull.scale_operator_pd}

Wishart distribution scale matrix as an OperatorPD.


- - -

#### `tf.contrib.distributions.WishartFull.std(name='std')` {#WishartFull.std}

Standard deviation of the Wishart distribution.


- - -

#### `tf.contrib.distributions.WishartFull.validate_args` {#WishartFull.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.WishartFull.variance(name='variance')` {#WishartFull.variance}

Variance of the Wishart distribution.

This function should not be confused with the covariance of the Wishart. The
covariance matrix would have shape `q x q` where,
`q = dimension * (dimension+1) / 2`
and having elements corresponding to some mapping from a lower-triangular
matrix to a vector-space.

This function returns the diagonal of the Covariance matrix but shaped
as a `dimension x dimension` matrix.

##### Args:


*  <b>`name`</b>: The name of this op.

##### Returns:


*  <b>`variance`</b>: `Tensor` of dtype `self.dtype`.


