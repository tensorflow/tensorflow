Base class for continuous probability distributions.

`ContinuousDistribution` defines the API for the likelihood functions `pdf`
and `log_pdf` of continuous probability distributions, and a property
`is_reparameterized` (returning `True` or `False`) which describes
whether the samples of this distribution are calculated in a differentiable
way from a non-parameterized distribution.  For example, the `Normal`
distribution with parameters `mu` and `sigma` is reparameterized as

```Normal(mu, sigma) = sigma * Normal(0, 1) + mu```

Subclasses must override `pdf` and `log_pdf` but one can call this base
class's implementation.  They must also override the `is_reparameterized`
property.

See `BaseDistribution` for more information on the API for probability
distributions.
- - -

#### `tf.contrib.distributions.ContinuousDistribution.batch_shape(name='batch_shape')` {#ContinuousDistribution.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.ContinuousDistribution.cdf(value, name='cdf')` {#ContinuousDistribution.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.dtype` {#ContinuousDistribution.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.entropy(name='entropy')` {#ContinuousDistribution.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.event_shape(name='event_shape')` {#ContinuousDistribution.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.ContinuousDistribution.get_batch_shape()` {#ContinuousDistribution.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.get_event_shape()` {#ContinuousDistribution.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.is_reparameterized` {#ContinuousDistribution.is_reparameterized}




- - -

#### `tf.contrib.distributions.ContinuousDistribution.log_cdf(value, name='log_cdf')` {#ContinuousDistribution.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.log_likelihood(value, name='log_likelihood')` {#ContinuousDistribution.log_likelihood}

Log likelihood of this distribution (same as log_pdf).


- - -

#### `tf.contrib.distributions.ContinuousDistribution.log_pdf(value, name='log_pdf')` {#ContinuousDistribution.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.mean(name='mean')` {#ContinuousDistribution.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.mode(name='mode')` {#ContinuousDistribution.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.name` {#ContinuousDistribution.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.pdf(value, name='pdf')` {#ContinuousDistribution.pdf}

Probability density function.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.sample(n, seed=None, name='sample')` {#ContinuousDistribution.sample}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.std(name='std')` {#ContinuousDistribution.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.ContinuousDistribution.variance(name='variance')` {#ContinuousDistribution.variance}

Variance of the distribution.


