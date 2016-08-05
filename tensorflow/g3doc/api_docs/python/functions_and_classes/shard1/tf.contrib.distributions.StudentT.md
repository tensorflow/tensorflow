Student's t distribution with degree-of-freedom parameter df.

#### Mathematical details

The PDF of this distribution is:

`f(t) = gamma((df+1)/2)/sqrt(df*pi)/gamma(df/2)*(1+t^2/df)^(-(df+1)/2)`

#### Examples

Examples of initialization of one or a batch of distributions.

```python
# Define a single scalar Student t distribution.
single_dist = tf.contrib.distributions.StudentT(df=3)

# Evaluate the pdf at 1, returning a scalar Tensor.
single_dist.pdf(1.)

# Define a batch of two scalar valued Student t's.
# The first has degrees of freedom 2, mean 1, and scale 11.
# The second 3, 2 and 22.
multi_dist = tf.contrib.distributions.StudentT(df=[2, 3],
                                               mu=[1, 2.],
                                               sigma=[11, 22.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
multi_dist.pdf([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
multi_dist.sample(3)
```

Arguments are broadcast when possible.

```python
# Define a batch of two Student's t distributions.
# Both have df 2 and mean 1, but different scales.
dist = tf.contrib.distributions.StudentT(df=2, mu=1, sigma=[11, 22.])

# Evaluate the pdf of both distributions on the same point, 3.0,
# returning a length 2 tensor.
dist.pdf(3.0)
```
- - -

#### `tf.contrib.distributions.StudentT.__init__(df, mu, sigma, validate_args=True, allow_nan_stats=False, name='StudentT')` {#StudentT.__init__}

Construct Student's t distributions.

The distributions have degree of freedom `df`, mean `mu`, and scale `sigma`.

The parameters `df`, `mu`, and `sigma` must be shaped in a way that supports
broadcasting (e.g. `df + mu + sigma` is a valid operation).

##### Args:


*  <b>`df`</b>: Floating point tensor, the degrees of freedom of the
    distribution(s). `df` must contain only positive values.
*  <b>`mu`</b>: Floating point tensor, the means of the distribution(s).
*  <b>`sigma`</b>: Floating point tensor, the scaling factor for the
    distribution(s). `sigma` must contain only positive values.
    Note that `sigma` is not the standard deviation of this distribution.
*  <b>`validate_args`</b>: Whether to assert that `df > 0, sigma > 0`. If
    `validate_args` is `False` and inputs are invalid, correct behavior is
    not guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: if mu and sigma are different dtypes.


- - -

#### `tf.contrib.distributions.StudentT.allow_nan_stats` {#StudentT.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.StudentT.batch_shape(name='batch_shape')` {#StudentT.batch_shape}




- - -

#### `tf.contrib.distributions.StudentT.cdf(value, name='cdf')` {#StudentT.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.StudentT.df` {#StudentT.df}

Degrees of freedom in these Student's t distribution(s).


- - -

#### `tf.contrib.distributions.StudentT.dtype` {#StudentT.dtype}




- - -

#### `tf.contrib.distributions.StudentT.entropy(name='entropy')` {#StudentT.entropy}

The entropy of Student t distribution(s).

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.StudentT.event_shape(name='event_shape')` {#StudentT.event_shape}




- - -

#### `tf.contrib.distributions.StudentT.get_batch_shape()` {#StudentT.get_batch_shape}




- - -

#### `tf.contrib.distributions.StudentT.get_event_shape()` {#StudentT.get_event_shape}




- - -

#### `tf.contrib.distributions.StudentT.is_continuous` {#StudentT.is_continuous}




- - -

#### `tf.contrib.distributions.StudentT.is_reparameterized` {#StudentT.is_reparameterized}




- - -

#### `tf.contrib.distributions.StudentT.log_cdf(value, name='log_cdf')` {#StudentT.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.StudentT.log_pdf(value, name='log_pdf')` {#StudentT.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.StudentT.log_pmf(value, name='log_pmf')` {#StudentT.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.StudentT.log_prob(x, name='log_prob')` {#StudentT.log_prob}

Log prob of observations in `x` under these Student's t-distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `df`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.StudentT.mean(name='mean')` {#StudentT.mean}

Mean of the distribution.

The mean of Student's T equals `mu` if `df > 1`, otherwise it is `NaN`.  If
`self.allow_nan_stats=False`, then an exception will be raised rather than
returning `NaN`.

##### Args:


*  <b>`name`</b>: A name to give this op.

##### Returns:

  The mean for every batch member, a `Tensor` with same `dtype` as self.


- - -

#### `tf.contrib.distributions.StudentT.mode(name='mode')` {#StudentT.mode}




- - -

#### `tf.contrib.distributions.StudentT.mu` {#StudentT.mu}

Locations of these Student's t distribution(s).


- - -

#### `tf.contrib.distributions.StudentT.name` {#StudentT.name}




- - -

#### `tf.contrib.distributions.StudentT.pdf(value, name='pdf')` {#StudentT.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.StudentT.pmf(value, name='pmf')` {#StudentT.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.StudentT.prob(x, name='prob')` {#StudentT.prob}

The PDF of observations in `x` under these Student's t distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `df`, `mu`, and
    `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the prob values of `x`.


- - -

#### `tf.contrib.distributions.StudentT.sample(sample_shape=(), seed=None, name='sample')` {#StudentT.sample}

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

#### `tf.contrib.distributions.StudentT.sample_n(n, seed=None, name='sample_n')` {#StudentT.sample_n}

Sample `n` observations from the Student t Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentT.sigma` {#StudentT.sigma}

Scaling factors of these Student's t distribution(s).


- - -

#### `tf.contrib.distributions.StudentT.std(name='std')` {#StudentT.std}




- - -

#### `tf.contrib.distributions.StudentT.validate_args` {#StudentT.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.StudentT.variance(name='variance')` {#StudentT.variance}

Variance of the distribution.

Variance for Student's T equals

```
df / (df - 2), when df > 2
infinity, when 1 < df <= 2
NaN, when df <= 1
```

The NaN state occurs because mean is undefined for `df <= 1`, and if
`self.allow_nan_stats` is `False`, an exception will be raised if any batch
members fall into this state.

##### Args:


*  <b>`name`</b>: A name for this op.

##### Returns:

  The variance for every batch member, a `Tensor` with same `dtype` as self.


