Bernoulli distribution.

The Bernoulli distribution is parameterized by p, the probability of a
positive event.
- - -

#### `tf.contrib.distributions.Bernoulli.__init__(logits=None, p=None, dtype=tf.int32, validate_args=True, allow_nan_stats=False, name='Bernoulli')` {#Bernoulli.__init__}

Construct Bernoulli distributions.

##### Args:


*  <b>`logits`</b>: An N-D `Tensor` representing the log-odds
    of a positive event. Each entry in the `Tensor` parametrizes
    an independent Bernoulli distribution where the probability of an event
    is sigmoid(logits).
*  <b>`p`</b>: An N-D `Tensor` representing the probability of a positive
      event. Each entry in the `Tensor` parameterizes an independent
      Bernoulli distribution.
*  <b>`dtype`</b>: dtype for samples.
*  <b>`validate_args`</b>: Whether to assert that `0 <= p <= 1`. If not validate_args,
   `log_pmf` may return nans.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: A name for this distribution.

##### Raises:


*  <b>`ValueError`</b>: If p and logits are passed, or if neither are passed.


- - -

#### `tf.contrib.distributions.Bernoulli.allow_nan_stats` {#Bernoulli.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Bernoulli.batch_shape(name='batch_shape')` {#Bernoulli.batch_shape}




- - -

#### `tf.contrib.distributions.Bernoulli.cdf(value, name='cdf')` {#Bernoulli.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.Bernoulli.dtype` {#Bernoulli.dtype}




- - -

#### `tf.contrib.distributions.Bernoulli.entropy(name='entropy')` {#Bernoulli.entropy}

Entropy of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`entropy`</b>: `Tensor` of the same type and shape as `p`.


- - -

#### `tf.contrib.distributions.Bernoulli.event_shape(name='event_shape')` {#Bernoulli.event_shape}




- - -

#### `tf.contrib.distributions.Bernoulli.get_batch_shape()` {#Bernoulli.get_batch_shape}




- - -

#### `tf.contrib.distributions.Bernoulli.get_event_shape()` {#Bernoulli.get_event_shape}




- - -

#### `tf.contrib.distributions.Bernoulli.is_continuous` {#Bernoulli.is_continuous}




- - -

#### `tf.contrib.distributions.Bernoulli.is_reparameterized` {#Bernoulli.is_reparameterized}




- - -

#### `tf.contrib.distributions.Bernoulli.log_cdf(value, name='log_cdf')` {#Bernoulli.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.Bernoulli.log_pdf(value, name='log_pdf')` {#Bernoulli.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Bernoulli.log_pmf(value, name='log_pmf')` {#Bernoulli.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Bernoulli.log_prob(event, name='log_prob')` {#Bernoulli.log_prob}

Log of the probability mass function.

##### Args:


*  <b>`event`</b>: `int32` or `int64` binary Tensor.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  The log-probabilities of the events.


- - -

#### `tf.contrib.distributions.Bernoulli.logits` {#Bernoulli.logits}




- - -

#### `tf.contrib.distributions.Bernoulli.mean(name='mean')` {#Bernoulli.mean}

Mean of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`mean`</b>: `Tensor` of the same type and shape as `p`.


- - -

#### `tf.contrib.distributions.Bernoulli.mode(name='mode')` {#Bernoulli.mode}

Mode of the distribution.

1 if p > 1-p. 0 otherwise.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`mode`</b>: binary `Tensor` of type self.dtype.


- - -

#### `tf.contrib.distributions.Bernoulli.name` {#Bernoulli.name}




- - -

#### `tf.contrib.distributions.Bernoulli.p` {#Bernoulli.p}




- - -

#### `tf.contrib.distributions.Bernoulli.pdf(value, name='pdf')` {#Bernoulli.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Bernoulli.pmf(value, name='pmf')` {#Bernoulli.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Bernoulli.prob(event, name='prob')` {#Bernoulli.prob}

Probability mass function.

##### Args:


*  <b>`event`</b>: `int32` or `int64` binary Tensor; must be broadcastable with `p`.
*  <b>`name`</b>: A name for this operation.

##### Returns:

  The probabilities of the events.


- - -

#### `tf.contrib.distributions.Bernoulli.q` {#Bernoulli.q}

1-p.


- - -

#### `tf.contrib.distributions.Bernoulli.sample(sample_shape=(), seed=None, name='sample')` {#Bernoulli.sample}

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

#### `tf.contrib.distributions.Bernoulli.sample_n(n, seed=None, name='sample_n')` {#Bernoulli.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar.  Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG.
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape` with values of type
      `self.dtype`.


- - -

#### `tf.contrib.distributions.Bernoulli.std(name='std')` {#Bernoulli.std}

Standard deviation of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`std`</b>: `Tensor` of the same type and shape as `p`.


- - -

#### `tf.contrib.distributions.Bernoulli.validate_args` {#Bernoulli.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Bernoulli.variance(name='variance')` {#Bernoulli.variance}

Variance of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`variance`</b>: `Tensor` of the same type and shape as `p`.


