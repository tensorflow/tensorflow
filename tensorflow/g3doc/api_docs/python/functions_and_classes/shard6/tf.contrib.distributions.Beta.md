Beta distribution.

This distribution is parameterized by `a` and `b` which are shape
parameters.

#### Mathematical details

The Beta is a distribution over the interval (0, 1).
The distribution has hyperparameters `a` and `b` and
probability mass function (pdf):

```pdf(x) = 1 / Beta(a, b) * x^(a - 1) * (1 - x)^(b - 1)```

where `Beta(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)`
is the beta function.


This class provides methods to create indexed batches of Beta
distributions. One entry of the broacasted
shape represents of `a` and `b` represents one single Beta distribution.
When calling distribution functions (e.g. `dist.pdf(x)`), `a`, `b`
and `x` are broadcast to the same shape (if possible).
Every entry in a/b/x corresponds to a single Beta distribution.

#### Examples

Creates 3 distributions.
The distribution functions can be evaluated on x.

```python
a = [1, 2, 3]
b = [1, 2, 3]
dist = Beta(a, b)
```

```python
# x same shape as a.
x = [.2, .3, .7]
dist.pdf(x)  # Shape [3]

# a/b will be broadcast to [[1, 2, 3], [1, 2, 3]] to match x.
x = [[.1, .4, .5], [.2, .3, .5]]
dist.pdf(x)  # Shape [2, 3]

# a/b will be broadcast to shape [5, 7, 3] to match x.
x = [[...]]  # Shape [5, 7, 3]
dist.pdf(x)  # Shape [5, 7, 3]
```

Creates a 2-batch of 3-class distributions.

```python
a = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
b = 5  # Shape []
dist = Beta(a, b)

# x will be broadcast to [[.2, .3, .9], [.2, .3, .9]] to match a/b.
x = [.2, .3, .9]
dist.pdf(x)  # Shape [2]
```
- - -

#### `tf.contrib.distributions.Beta.__init__(a, b, validate_args=True, allow_nan_stats=False, name='Beta')` {#Beta.__init__}

Initialize a batch of Beta distributions.

##### Args:


*  <b>`a`</b>: Positive floating point tensor with shape broadcastable to
    `[N1,..., Nm]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
     different Beta distributions. This also defines the
     dtype of the distribution.
*  <b>`b`</b>: Positive floating point tensor with shape broadcastable to
    `[N1,..., Nm]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
     different Beta distributions.
*  <b>`validate_args`</b>: Whether to assert valid values for parameters `a` and `b`,
    and `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prefix Ops created by this distribution class.


*  <b>`Examples`</b>: 

```python
# Define 1-batch.
dist = Beta(1.1, 2.0)

# Define a 2-batch.
dist = Beta([1.0, 2.0], [4.0, 5.0])
```


- - -

#### `tf.contrib.distributions.Beta.a` {#Beta.a}

Shape parameter.


- - -

#### `tf.contrib.distributions.Beta.allow_nan_stats` {#Beta.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Beta.b` {#Beta.b}

Shape parameter.


- - -

#### `tf.contrib.distributions.Beta.batch_shape(name='batch_shape')` {#Beta.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Beta.cdf(x, name='cdf')` {#Beta.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.Beta.dtype` {#Beta.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.Beta.entropy(name='entropy')` {#Beta.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.Beta.event_shape(name='event_shape')` {#Beta.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Beta.get_batch_shape()` {#Beta.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.Beta.get_event_shape()` {#Beta.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.Beta.is_continuous` {#Beta.is_continuous}




- - -

#### `tf.contrib.distributions.Beta.is_reparameterized` {#Beta.is_reparameterized}




- - -

#### `tf.contrib.distributions.Beta.log_cdf(x, name='log_cdf')` {#Beta.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.Beta.log_pdf(value, name='log_pdf')` {#Beta.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Beta.log_pmf(value, name='log_pmf')` {#Beta.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Beta.log_prob(x, name='log_prob')` {#Beta.log_prob}

`Log(P[counts])`, computed for every batch member.

##### Args:


*  <b>`x`</b>: Non-negative floating point tensor whose shape can
    be broadcast with `self.a` and `self.b`.  For fixed leading
    dimensions, the last dimension represents counts for the corresponding
    Beta distribution in `self.a` and `self.b`. `x` is only legal if
    0 < x < 1.
*  <b>`name`</b>: Name to give this Op, defaults to "log_prob".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Beta.mean(name='mean')` {#Beta.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.Beta.mode(name='mode')` {#Beta.mode}

Mode of the distribution.

Note that the mode for the Beta distribution is only defined
when `a > 1`, `b > 1`. This returns the mode when `a > 1` and `b > 1`,
and NaN otherwise. If `self.allow_nan_stats` is `False`, an exception
will be raised rather than returning `NaN`.

##### Args:


*  <b>`name`</b>: The name for this op.

##### Returns:

  Mode of the Beta distribution.


- - -

#### `tf.contrib.distributions.Beta.name` {#Beta.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.Beta.pdf(value, name='pdf')` {#Beta.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Beta.pmf(value, name='pmf')` {#Beta.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Beta.prob(x, name='prob')` {#Beta.prob}

`P[x]`, computed for every batch member.

##### Args:


*  <b>`x`</b>: Non-negative floating point tensor whose shape can
    be broadcast with `self.a` and `self.b`.  For fixed leading
    dimensions, the last dimension represents x for the corresponding Beta
    distribution in `self.a` and `self.b`. `x` is only legal if is
    between 0 and 1.
*  <b>`name`</b>: Name to give this Op, defaults to "pdf".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Beta.sample(sample_shape=(), seed=None, name='sample')` {#Beta.sample}

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

#### `tf.contrib.distributions.Beta.sample_n(n, seed=None, name='sample_n')` {#Beta.sample_n}

Sample `n` observations from the Beta Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.Beta.std(name='std')` {#Beta.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.Beta.validate_args` {#Beta.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Beta.variance(name='variance')` {#Beta.variance}

Variance of the distribution.


