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

#### `tf.contrib.distributions.Binomial.__init__(n, logits=None, p=None, validate_args=True, allow_nan_stats=False, name='Binomial')` {#Binomial.__init__}

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
*  <b>`validate_args`</b>: Whether to assert valid values for parameters `n` and `p`,
    and `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
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

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Binomial.batch_shape(name='batch_shape')` {#Binomial.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Binomial.cdf(value, name='cdf')` {#Binomial.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.Binomial.dtype` {#Binomial.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.Binomial.entropy(name='entropy')` {#Binomial.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.Binomial.event_shape(name='event_shape')` {#Binomial.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Binomial.get_batch_shape()` {#Binomial.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.Binomial.get_event_shape()` {#Binomial.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.Binomial.is_continuous` {#Binomial.is_continuous}




- - -

#### `tf.contrib.distributions.Binomial.is_reparameterized` {#Binomial.is_reparameterized}




- - -

#### `tf.contrib.distributions.Binomial.log_cdf(value, name='log_cdf')` {#Binomial.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.Binomial.log_pdf(value, name='log_pdf')` {#Binomial.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Binomial.log_pmf(value, name='log_pmf')` {#Binomial.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Binomial.log_prob(counts, name='log_prob')` {#Binomial.log_prob}

`Log(P[counts])`, computed for every batch member.

For each batch member of counts `k`, `P[counts]` is the probability that
after sampling `n` draws from this Binomial distribution, the number of
successes is `k`.  Note that different sequences of draws can result in the
same counts, thus the probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative tensor with dtype `dtype` and whose shape can be
    broadcast with `self.p` and `self.n`. `counts` is only legal if it is
    less than or equal to `n` and its components are equal to integer
    values.
*  <b>`name`</b>: Name to give this Op, defaults to "log_prob".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Binomial.logits` {#Binomial.logits}

Log-odds.


- - -

#### `tf.contrib.distributions.Binomial.mean(name='mean')` {#Binomial.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.Binomial.mode(name='mode')` {#Binomial.mode}

Mode of the distribution.

Note that when `(n + 1) * p` is an integer, there are actually two modes.
Namely, `(n + 1) * p` and `(n + 1) * p - 1` are both modes. Here we return
only the larger of the two modes.

##### Args:


*  <b>`name`</b>: The name for this op.

##### Returns:

  The mode of the Binomial distribution.


- - -

#### `tf.contrib.distributions.Binomial.n` {#Binomial.n}

Number of trials.


- - -

#### `tf.contrib.distributions.Binomial.name` {#Binomial.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.Binomial.p` {#Binomial.p}

Probability of success.


- - -

#### `tf.contrib.distributions.Binomial.pdf(value, name='pdf')` {#Binomial.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Binomial.pmf(value, name='pmf')` {#Binomial.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Binomial.prob(counts, name='prob')` {#Binomial.prob}

`P[counts]`, computed for every batch member.


For each batch member of counts `k`, `P[counts]` is the probability that
after sampling `n` draws from this Binomial distribution, the number of
successes is `k`.  Note that different sequences of draws can result in the
same counts, thus the probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative tensor with dtype `dtype` and whose shape can be
    broadcast with `self.p` and `self.n`. `counts` is only legal if it is
    less than or equal to `n` and its components are equal to integer
    values.
*  <b>`name`</b>: Name to give this Op, defaults to "prob".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Binomial.sample(sample_shape=(), seed=None, name='sample')` {#Binomial.sample}

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

#### `tf.contrib.distributions.Binomial.sample_n(n, seed=None, name='sample_n')` {#Binomial.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Binomial.std(name='std')` {#Binomial.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.Binomial.validate_args` {#Binomial.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Binomial.variance(name='variance')` {#Binomial.variance}

Variance of the distribution.


