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
#   in tf.contrib.distributions.batch_matrix_diag_transform
#   (operator_pd_cholesky.py).
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


