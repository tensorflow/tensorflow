The scalar Normal distribution with mean and stddev parameters mu, sigma.

#### Mathematical details

The PDF of this distribution is:

```f(x) = sqrt(1/(2*pi*sigma^2)) exp(-(x-mu)^2/(2*sigma^2))```

#### Examples

Examples of initialization of one or a batch of distributions.

```python
# Define a single scalar Normal distribution.
dist = tf.contrib.distributions.Normal(mu=0, sigma=3)

# Evaluate the cdf at 1, returning a scalar.
dist.cdf(1)

# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = tf.contrib.distributions.Normal(mu=[1, 2.], sigma=[11, 22.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
dist.pdf([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
dist.sample(3)
```

Arguments are broadcast when possible.

```python
# Define a batch of two scalar valued Normals.
# Both have mean 1, but different standard deviations.
dist = tf.contrib.distributions.Normal(mu=1, sigma=[11, 22.])

# Evaluate the pdf of both distributions on the same point, 3.0,
# returning a length 2 tensor.
dist.pdf(3.0)
```
- - -

#### `tf.contrib.distributions.Normal.__init__(mu, sigma, validate_args=True, allow_nan_stats=False, name='Normal')` {#Normal.__init__}

Construct Normal distributions with mean and stddev `mu` and `sigma`.

The parameters `mu` and `sigma` must be shaped in a way that supports
broadcasting (e.g. `mu + sigma` is a valid operation).

##### Args:


*  <b>`mu`</b>: Floating point tensor, the means of the distribution(s).
*  <b>`sigma`</b>: Floating point tensor, the stddevs of the distribution(s).
    sigma must contain only positive values.
*  <b>`validate_args`</b>: Whether to assert that `sigma > 0`. If `validate_args` is
    `False`, correct output is not guaranteed when input is invalid.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: if mu and sigma are different dtypes.


- - -

#### `tf.contrib.distributions.Normal.allow_nan_stats` {#Normal.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Normal.batch_shape(name='batch_shape')` {#Normal.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Normal.cdf(x, name='cdf')` {#Normal.cdf}

CDF of observations in `x` under these Normal distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.Normal.dtype` {#Normal.dtype}




- - -

#### `tf.contrib.distributions.Normal.entropy(name='entropy')` {#Normal.entropy}

The entropy of Normal distribution(s).

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Normal.event_shape(name='event_shape')` {#Normal.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Normal.get_batch_shape()` {#Normal.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.Normal.get_event_shape()` {#Normal.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.Normal.is_continuous` {#Normal.is_continuous}




- - -

#### `tf.contrib.distributions.Normal.is_reparameterized` {#Normal.is_reparameterized}




- - -

#### `tf.contrib.distributions.Normal.log_cdf(x, name='log_cdf')` {#Normal.log_cdf}

Log CDF of observations `x` under these Normal distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Normal.log_pdf(value, name='log_pdf')` {#Normal.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Normal.log_pmf(value, name='log_pmf')` {#Normal.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Normal.log_prob(x, name='log_prob')` {#Normal.log_prob}

Log prob of observations in `x` under these Normal distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.Normal.mean(name='mean')` {#Normal.mean}

Mean of this distribution.


- - -

#### `tf.contrib.distributions.Normal.mode(name='mode')` {#Normal.mode}

Mode of this distribution.


- - -

#### `tf.contrib.distributions.Normal.mu` {#Normal.mu}

Distribution parameter for the mean.


- - -

#### `tf.contrib.distributions.Normal.name` {#Normal.name}




- - -

#### `tf.contrib.distributions.Normal.pdf(value, name='pdf')` {#Normal.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Normal.pmf(value, name='pmf')` {#Normal.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Normal.prob(x, name='prob')` {#Normal.prob}

The PDF of observations in `x` under these Normal distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the prob values of `x`.


- - -

#### `tf.contrib.distributions.Normal.sample(sample_shape=(), seed=None, name='sample')` {#Normal.sample}

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

#### `tf.contrib.distributions.Normal.sample_n(n, seed=None, name='sample_n')` {#Normal.sample_n}

Sample `n` observations from the Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.Normal.sigma` {#Normal.sigma}

Distribution parameter for standard deviation.


- - -

#### `tf.contrib.distributions.Normal.std(name='std')` {#Normal.std}

Standard deviation of this distribution.


- - -

#### `tf.contrib.distributions.Normal.validate_args` {#Normal.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Normal.variance(name='variance')` {#Normal.variance}

Variance of this distribution.


