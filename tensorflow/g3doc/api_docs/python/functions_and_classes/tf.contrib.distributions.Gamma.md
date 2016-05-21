The `Gamma` distribution with parameter alpha and beta.

The parameters are the shape and inverse scale parameters alpha, beta.

The PDF of this distribution is:

```pdf(x) = (beta^alpha)(x^(alpha-1))e^(-x*beta)/Gamma(alpha), x > 0```

and the CDF of this distribution is:

```cdf(x) =  GammaInc(alpha, beta * x) / Gamma(alpha), x > 0```

where GammaInc is the incomplete lower Gamma function.

Examples:

```python
dist = Gamma(alpha=3.0, beta=2.0)
dist2 = Gamma(alpha=[3.0, 4.0], beta=[2.0, 3.0])
```
- - -

#### `tf.contrib.distributions.Gamma.__init__(alpha, beta, name='Gamma')` {#Gamma.__init__}

Construct Gamma distributions with parameters `alpha` and `beta`.

The parameters `alpha` and `beta` must be shaped in a way that supports
broadcasting (e.g. `alpha + beta` is a valid operation).

##### Args:


*  <b>`alpha`</b>: `float` or `double` tensor, the shape params of the
    distribution(s).
    alpha must contain only positive values.
*  <b>`beta`</b>: `float` or `double` tensor, the inverse scale params of the
    distribution(s).
    beta must contain only positive values.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: if `alpha` and `beta` are different dtypes.


- - -

#### `tf.contrib.distributions.Gamma.alpha` {#Gamma.alpha}




- - -

#### `tf.contrib.distributions.Gamma.batch_shape(name='batch_shape')` {#Gamma.batch_shape}




- - -

#### `tf.contrib.distributions.Gamma.beta` {#Gamma.beta}




- - -

#### `tf.contrib.distributions.Gamma.cdf(x, name='cdf')` {#Gamma.cdf}




- - -

#### `tf.contrib.distributions.Gamma.dtype` {#Gamma.dtype}




- - -

#### `tf.contrib.distributions.Gamma.entropy(name='entropy')` {#Gamma.entropy}

The entropy of Gamma distribution(s).

This is defined to be

```entropy = alpha - log(beta) + log(Gamma(alpha))
             + (1-alpha)digamma(alpha)```

where digamma(alpha) is the digamma function.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Gamma.event_shape(name='event_shape')` {#Gamma.event_shape}




- - -

#### `tf.contrib.distributions.Gamma.get_batch_shape()` {#Gamma.get_batch_shape}




- - -

#### `tf.contrib.distributions.Gamma.get_event_shape()` {#Gamma.get_event_shape}




- - -

#### `tf.contrib.distributions.Gamma.is_reparameterized` {#Gamma.is_reparameterized}




- - -

#### `tf.contrib.distributions.Gamma.log_cdf(x, name='log_cdf')` {#Gamma.log_cdf}

Log CDF of observations `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Gamma.log_pdf(x, name='log_pdf')` {#Gamma.log_pdf}

Log pdf of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.Gamma.mean` {#Gamma.mean}




- - -

#### `tf.contrib.distributions.Gamma.name` {#Gamma.name}




- - -

#### `tf.contrib.distributions.Gamma.pdf(x, name='pdf')` {#Gamma.pdf}




- - -

#### `tf.contrib.distributions.Gamma.sample(n, seed=None, name=None)` {#Gamma.sample}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Gamma.variance` {#Gamma.variance}




