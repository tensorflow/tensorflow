### `tf.contrib.distributions.normal_conjugates_known_scale_posterior(prior, scale, s, n)` {#normal_conjugates_known_scale_posterior}

Posterior Normal distribution with conjugate prior on the mean.

This model assumes that `n` observations (with sum `s`) come from a
Normal with unknown mean `loc` (described by the Normal `prior`)
and known variance `scale^2`.  The "known scale posterior" is
the distribution of the unknown `loc`.

Accepts a prior Normal distribution object, having parameters
`loc0` and `scale0`, as well as known `scale` values of the predictive
distribution(s) (also assumed Normal),
and statistical estimates `s` (the sum(s) of the observations) and
`n` (the number(s) of observations).

Returns a posterior (also Normal) distribution object, with parameters
`(loc', scale'^2)`, where:

```
mu ~ N(mu', sigma'^2)
sigma'^2 = 1/(1/sigma0^2 + n/sigma^2),
mu' = (mu0/sigma0^2 + s/sigma^2) * sigma'^2.
```

Distribution parameters from `prior`, as well as `scale`, `s`, and `n`.
will broadcast in the case of multidimensional sets of parameters.

##### Args:


*  <b>`prior`</b>: `Normal` object of type `dtype`:
    the prior distribution having parameters `(loc0, scale0)`.
*  <b>`scale`</b>: tensor of type `dtype`, taking values `scale > 0`.
    The known stddev parameter(s).
*  <b>`s`</b>: Tensor of type `dtype`.  The sum(s) of observations.
*  <b>`n`</b>: Tensor of type `int`.  The number(s) of observations.

##### Returns:

  A new Normal posterior distribution object for the unknown observation
  mean `loc`.

##### Raises:


*  <b>`TypeError`</b>: if dtype of `s` does not match `dtype`, or `prior` is not a
    Normal object.

