### `tf.contrib.distributions.gaussian_conjugates_known_sigma_posterior(prior, sigma, s, n)` {#gaussian_conjugates_known_sigma_posterior}

Posterior Gaussian distribution with conjugate prior on the mean.

This model assumes that `n` observations (with sum `s`) come from a
Gaussian with unknown mean `mu` (described by the Gaussian `prior`)
and known variance `sigma^2`.  The "known sigma posterior" is
the distribution of the unknown `mu`.

Accepts a prior Gaussian distribution object, having parameters
`mu0` and `sigma0`, as well as known `sigma` values of the predictive
distribution(s) (also assumed Gaussian),
and statistical estimates `s` (the sum(s) of the observations) and
`n` (the number(s) of observations).

Returns a posterior (also Gaussian) distribution object, with parameters
`(mu', sigma'^2)`, where:

```
mu ~ N(mu', sigma'^2)
sigma'^2 = 1/(1/sigma0^2 + n/sigma^2),
mu' = (mu0/sigma0^2 + s/sigma^2) * sigma'^2.
```

Distribution parameters from `prior`, as well as `sigma`, `s`, and `n`.
will broadcast in the case of multidimensional sets of parameters.

##### Args:


*  <b>`prior`</b>: `Gaussian` object of type `dtype`:
    the prior distribution having parameters `(mu0, sigma0)`.
*  <b>`sigma`</b>: tensor of type `dtype`, taking values `sigma > 0`.
    The known stddev parameter(s).
*  <b>`s`</b>: Tensor of type `dtype`.  The sum(s) of observations.
*  <b>`n`</b>: Tensor of type `int`.  The number(s) of observations.

##### Returns:

  A new Gaussian posterior distribution object for the unknown observation
  mean `mu`.

##### Raises:


*  <b>`TypeError`</b>: if dtype of `s` does not match `dtype`, or `prior` is not a
    Gaussian object.

