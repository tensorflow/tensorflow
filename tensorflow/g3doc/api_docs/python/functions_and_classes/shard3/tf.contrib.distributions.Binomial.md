Binomial distribution.

This distribution is parameterized by a vector `p` of probabilities and `n`,
the total counts.

#### Mathematical details

The Binomial is a distribution over the number of successes in `n` independent
trials, with each trial having the same probability of success `p`.
The probability mass function (pmf):

```pmf(k) = n! / (k! * (n - k)!) * (p)^k * (1 - p)^(n - k)```

#### Examples

Create a single distribution, corresponding to 5 coin flips.

```python
dist = Binomial(n=5., p=.5)
```

Create a single distribution (using logits), corresponding to 5 coin flips.

```python
dist = Binomial(n=5., logits=0.)
```

Creates 3 distributions with the third distribution most likely to have
successes.

```python
p = [.2, .3, .8]
# n will be broadcast to [4., 4., 4.], to match p.
dist = Binomial(n=4., p=p)
```

The distribution functions can be evaluated on counts.

```python
# counts same shape as p.
counts = [1., 2, 3]
dist.prob(counts)  # Shape [3]

# p will be broadcast to [[.2, .3, .8], [.2, .3, .8]] to match counts.
counts = [[1., 2, 1], [2, 2, 4]]
dist.prob(counts)  # Shape [2, 3]

# p will be broadcast to shape [5, 7, 3] to match counts.
counts = [[...]]  # Shape [5, 7, 3]
dist.prob(counts)  # Shape [5, 7, 3]
```
- - -

#### `tf.contrib.distributions.Binomial.__init__(n, logits=None, p=None, validate_args=False, allow_nan_stats=True, name='Binomial')` {#Binomial.__init__}

Initialize a batch of Binomial distributions.

##### Args:


*  <b>`n`</b>: Non-negative floating point tensor with shape broadcastable to
    `[N1,..., Nm]` with `m >= 0` and the same dtype as `p` or `logits`.
    Defines this as a batch of `N1 x ... x Nm` different Binomial
    distributions. Its components should be equal to integer values.
*  <b>`logits`</b>: Floating point tensor representing the log-odds of a
    positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
    the same dtype as `n`. Each entry represents logits for the probability
    of success for independent Binomial distributions.
*  <b>`p`</b>: Positive floating point tensor with shape broadcastable to
    `[N1,..., Nm]` `m >= 0`, `p in [0, 1]`. Each entry represents the
    probability of success for independent Binomial distributions.
*  <b>`validate_args`</b>: `Boolean`, default `False`.  Whether to assert valid values
    for parameters `n`, `p`, and `x` in `prob` and `log_prob`.
    If `False` and inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: `Boolean`, default `True`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prefix Ops created by this distribution class.


*  <b>`Examples`</b>: 

```python
# Define 1-batch of a binomial distribution.
dist = Binomial(n=2., p=.9)

# Define a 2-batch.
dist = Binomial(n=[4., 5], p=[.1, .3])
```


- - -

#### `tf.contrib.distributions.Binomial.allow_nan_stats` {#Binomial.allow_nan_stats}

Python boolean describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense.  E.g., the variance
of a Cauchy distribution is infinity.  However, sometimes the
statistic is undefined, e.g., if a distribution's pdf does not achieve a
maximum within the support of the distribution, the mode is undefined.
If the mean is undefined, then by definition the variance is undefined.
E.g. the mean for Student's T for df = 1 is undefined (no clear way to say
it is either + or - infinity), so the variance = E[(X - mean)^2] is also
undefined.

##### Returns:


*  <b>`allow_nan_stats`</b>: Python boolean.


- - -

#### `tf.contrib.distributions.Binomial.batch_shape(name='batch_shape')` {#Binomial.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Binomial.cdf(value, name='cdf')` {#Binomial.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.dtype` {#Binomial.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.Binomial.entropy(name='entropy')` {#Binomial.entropy}

Shanon entropy in nats.


- - -

#### `tf.contrib.distributions.Binomial.event_shape(name='event_shape')` {#Binomial.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Binomial.get_batch_shape()` {#Binomial.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Binomial.get_event_shape()` {#Binomial.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Binomial.is_continuous` {#Binomial.is_continuous}




- - -

#### `tf.contrib.distributions.Binomial.is_reparameterized` {#Binomial.is_reparameterized}




- - -

#### `tf.contrib.distributions.Binomial.log_cdf(value, name='log_cdf')` {#Binomial.log_cdf}

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.log_pdf(value, name='log_pdf')` {#Binomial.log_pdf}

Log probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.Binomial.log_pmf(value, name='log_pmf')` {#Binomial.log_pmf}

Log probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.Binomial.log_prob(value, name='log_prob')` {#Binomial.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.log_survival_function(value, name='log_survival_function')` {#Binomial.log_survival_function}

Log survival function.

Given random variable `X`, the survival function is defined:

```
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.logits` {#Binomial.logits}

Log-odds.


- - -

#### `tf.contrib.distributions.Binomial.mean(name='mean')` {#Binomial.mean}

Mean.


- - -

#### `tf.contrib.distributions.Binomial.mode(name='mode')` {#Binomial.mode}

Mode.


- - -

#### `tf.contrib.distributions.Binomial.n` {#Binomial.n}

Number of trials.


- - -

#### `tf.contrib.distributions.Binomial.name` {#Binomial.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.Binomial.p` {#Binomial.p}

Probability of success.


- - -

#### `tf.contrib.distributions.Binomial.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Binomial.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.Binomial.param_static_shapes(cls, sample_shape)` {#Binomial.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.Binomial.parameters` {#Binomial.parameters}

Dictionary of parameters used by this `Distribution`.


- - -

#### `tf.contrib.distributions.Binomial.pdf(value, name='pdf')` {#Binomial.pdf}

Probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.Binomial.pmf(value, name='pmf')` {#Binomial.pmf}

Probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.Binomial.prob(value, name='prob')` {#Binomial.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.sample(sample_shape=(), seed=None, name='sample')` {#Binomial.sample}

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

##### Args:


*  <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.


- - -

#### `tf.contrib.distributions.Binomial.sample_n(n, seed=None, name='sample_n')` {#Binomial.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: `Scalar` `Tensor` of type `int32` or `int64`, the number of
    observations to sample.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with a prepended dimension (n,).

##### Raises:


*  <b>`TypeError`</b>: if `n` is not an integer type.


- - -

#### `tf.contrib.distributions.Binomial.std(name='std')` {#Binomial.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.Binomial.survival_function(value, name='survival_function')` {#Binomial.survival_function}

Survival function.

Given random variable `X`, the survival function is defined:

```
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.validate_args` {#Binomial.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.Binomial.variance(name='variance')` {#Binomial.variance}

Variance.


