Base class for discrete probability distributions.

`DiscreteDistribution` defines the API for the likelihood functions `pmf` and
`log_pmf` of discrete probability distributions.

Subclasses must override both `pmf` and `log_pmf` but one can call this base
class's implementation.

See `BaseDistribution` for more information on the API for probability
distributions.
- - -

#### `tf.contrib.distributions.DiscreteDistribution.batch_shape(name='batch_shape')` {#DiscreteDistribution.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.DiscreteDistribution.cdf(value, name='cdf')` {#DiscreteDistribution.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.dtype` {#DiscreteDistribution.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.entropy(name='entropy')` {#DiscreteDistribution.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.event_shape(name='event_shape')` {#DiscreteDistribution.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.DiscreteDistribution.get_batch_shape()` {#DiscreteDistribution.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.get_event_shape()` {#DiscreteDistribution.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.log_cdf(value, name='log_cdf')` {#DiscreteDistribution.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.log_likelihood(value, name='log_likelihood')` {#DiscreteDistribution.log_likelihood}

Log likelihood of this distribution (same as log_pmf).


- - -

#### `tf.contrib.distributions.DiscreteDistribution.log_pmf(value, name='log_pmf')` {#DiscreteDistribution.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.mean(name='mean')` {#DiscreteDistribution.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.mode(name='mode')` {#DiscreteDistribution.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.name` {#DiscreteDistribution.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.pmf(value, name='pmf')` {#DiscreteDistribution.pmf}

Probability mass function.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.sample(n, seed=None, name='sample')` {#DiscreteDistribution.sample}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.std(name='std')` {#DiscreteDistribution.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.DiscreteDistribution.variance(name='variance')` {#DiscreteDistribution.variance}

Variance of the distribution.


