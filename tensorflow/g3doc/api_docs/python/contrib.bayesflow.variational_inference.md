<!-- This file is machine generated: DO NOT EDIT! -->

# BayesFlow Variational Inference (contrib)
[TOC]

Variational inference.

- - -

### `tf.contrib.bayesflow.variational_inference.elbo(log_likelihood, variational_with_prior=None, keep_batch_dim=True, form=None, name='ELBO')` {#elbo}

Evidence Lower BOund. `log p(x) >= ELBO`.

Optimization objective for inference of hidden variables by variational
inference.

This function is meant to be used in conjunction with `DistributionTensor`.
The user should build out the inference network, using `DistributionTensor`s
as latent variables, and the generative network. `elbo` at minimum needs
`p(x|Z)` and assumes that all `DistributionTensor`s upstream of `p(x|Z)` are
the variational distributions. Use `register_prior` to register `Distribution`
priors for each `DistributionTensor`. Alternatively, pass in
`variational_with_prior` specifying all variational distributions and their
priors.

Mathematical details:

```
log p(x) =  log \int p(x, Z) dZ
         =  log \int \frac {q(Z)p(x, Z)}{q(Z)} dZ
         =  log E_q[\frac {p(x, Z)}{q(Z)}]
         >= E_q[log \frac {p(x, Z)}{q(Z)}] = L[q; p, x]  # ELBO

L[q; p, x] = E_q[log p(x|Z)p(Z)] - E_q[log q(Z)]
           = E_q[log p(x|Z)p(Z)] + H[q]           (1)
           = E_q[log p(x|Z)] - KL(q || p)         (2)

H - Entropy
KL - Kullback-Leibler divergence
```

See section 2.2 of Stochastic Variational Inference by Hoffman et al. for
more, including the ELBO's equivalence to minimizing `KL(q(Z)||p(Z|x))`
in the fully Bayesian setting. https://arxiv.org/pdf/1206.7051.pdf.

`form` specifies which form of the ELBO is used. `form=ELBOForms.default`
tries, in order of preference: analytic KL, analytic entropy, sampling.

Multiple entries in the `variational_with_prior` dict implies a factorization.
e.g. `q(Z) = q(z1)q(z2)q(z3)`.

##### Args:


*  <b>`log_likelihood`</b>: `Tensor` log p(x|Z).
*  <b>`variational_with_prior`</b>: dict from `DistributionTensor` q(Z) to
    `Distribution` p(Z). If `None`, defaults to all `DistributionTensor`
    objects upstream of `log_likelihood` with priors registered with
    `register_prior`.
*  <b>`keep_batch_dim`</b>: bool. Whether to keep the batch dimension when summing
    entropy/KL term. When the sample is per data point, this should be True;
    otherwise (e.g. in a Bayesian NN), this should be False.
*  <b>`form`</b>: ELBOForms constant. Controls how the ELBO is computed. Defaults to
    ELBOForms.default.
*  <b>`name`</b>: name to prefix ops with.

##### Returns:

  `Tensor` ELBO of the same type and shape as `log_likelihood`.

##### Raises:


*  <b>`TypeError`</b>: if variationals in `variational_with_prior` are not
    `DistributionTensor`s or if priors are not `BaseDistribution`s.
*  <b>`TypeError`</b>: if form is not a valid ELBOForms constant.
*  <b>`ValueError`</b>: if `variational_with_prior` is None and there are no
    `DistributionTensor`s upstream of `log_likelihood`.
*  <b>`ValueError`</b>: if any variational does not have a prior passed or registered.


- - -

### `tf.contrib.bayesflow.variational_inference.elbo_with_log_joint(log_joint, variational=None, keep_batch_dim=True, form=None, name='ELBO')` {#elbo_with_log_joint}

Evidence Lower BOund. `log p(x) >= ELBO`.

This method is for models that have computed `p(x,Z)` instead of `p(x|Z)`.
See `elbo` for further details.

Because only the joint is specified, analytic KL is not available.

##### Args:


*  <b>`log_joint`</b>: `Tensor` log p(x, Z).
*  <b>`variational`</b>: list of `DistributionTensor` q(Z). If `None`, defaults to all
    `DistributionTensor` objects upstream of `log_joint`.
*  <b>`keep_batch_dim`</b>: bool. Whether to keep the batch dimension when summing
    entropy term. When the sample is per data point, this should be True;
    otherwise (e.g. in a Bayesian NN), this should be False.
*  <b>`form`</b>: ELBOForms constant. Controls how the ELBO is computed. Defaults to
    ELBOForms.default.
*  <b>`name`</b>: name to prefix ops with.

##### Returns:

  `Tensor` ELBO of the same type and shape as `log_joint`.

##### Raises:


*  <b>`TypeError`</b>: if variationals in `variational` are not `DistributionTensor`s.
*  <b>`TypeError`</b>: if form is not a valid ELBOForms constant.
*  <b>`ValueError`</b>: if `variational` is None and there are no `DistributionTensor`s
    upstream of `log_joint`.
*  <b>`ValueError`</b>: if form is ELBOForms.analytic_kl.


- - -

### `class tf.contrib.bayesflow.variational_inference.ELBOForms` {#ELBOForms}

Constants to control the `elbo` calculation.

`analytic_kl` uses the analytic KL divergence between the
variational distribution(s) and the prior(s).

`analytic_entropy` uses the analytic entropy of the variational
distribution(s).

`sample` uses the sample KL or the sample entropy is the joint is provided.

See `elbo` for what is used with `default`.
- - -

#### `tf.contrib.bayesflow.variational_inference.ELBOForms.check_form(form)` {#ELBOForms.check_form}





- - -

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


