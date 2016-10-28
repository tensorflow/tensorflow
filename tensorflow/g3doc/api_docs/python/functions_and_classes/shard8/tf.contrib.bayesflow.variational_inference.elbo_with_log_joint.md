### `tf.contrib.bayesflow.variational_inference.elbo_with_log_joint(log_joint, variational=None, keep_batch_dim=True, form=None, name='ELBO')` {#elbo_with_log_joint}

Evidence Lower BOund. `log p(x) >= ELBO`.

This method is for models that have computed `p(x,Z)` instead of `p(x|Z)`.
See `elbo` for further details.

Because only the joint is specified, analytic KL is not available.

##### Args:


*  <b>`log_joint`</b>: `Tensor` log p(x, Z).
*  <b>`variational`</b>: list of `StochasticTensor` q(Z). If `None`, defaults to all
    `StochasticTensor` objects upstream of `log_joint`.
*  <b>`keep_batch_dim`</b>: bool. Whether to keep the batch dimension when summing
    entropy term. When the sample is per data point, this should be True;
    otherwise (e.g. in a Bayesian NN), this should be False.
*  <b>`form`</b>: ELBOForms constant. Controls how the ELBO is computed. Defaults to
    ELBOForms.default.
*  <b>`name`</b>: name to prefix ops with.

##### Returns:

  `Tensor` ELBO of the same type and shape as `log_joint`.

##### Raises:


*  <b>`TypeError`</b>: if variationals in `variational` are not `StochasticTensor`s.
*  <b>`TypeError`</b>: if form is not a valid ELBOForms constant.
*  <b>`ValueError`</b>: if `variational` is None and there are no `StochasticTensor`s
    upstream of `log_joint`.
*  <b>`ValueError`</b>: if form is ELBOForms.analytic_kl.

