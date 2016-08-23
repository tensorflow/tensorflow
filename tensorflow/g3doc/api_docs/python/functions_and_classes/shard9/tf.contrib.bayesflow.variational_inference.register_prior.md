### `tf.contrib.bayesflow.variational_inference.register_prior(variational, prior)` {#register_prior}

Associate a variational `DistributionTensor` with a `Distribution` prior.

This is a helper function used in conjunction with `elbo` that allows users
to specify the mapping between variational distributions and their priors
without having to pass in `variational_with_prior` explicitly.

##### Args:


*  <b>`variational`</b>: `DistributionTensor` q(Z). Approximating distribution.
*  <b>`prior`</b>: `Distribution` p(Z). Prior distribution.

##### Returns:

  None

##### Raises:


*  <b>`ValueError`</b>: if variational is not a `DistributionTensor` or `prior` is not
    a `Distribution`.

