### `tf.contrib.bayesflow.variational_inference.elbo(log_likelihood, variational_with_prior=None, keep_batch_dim=True, form=None, name='ELBO')` {#elbo}

Evidence Lower BOund. `log p(x) >= ELBO`.

Optimization objective for inference of hidden variables by variational
inference.

This function is meant to be used in conjunction with `StochasticTensor`.
The user should build out the inference network, using `StochasticTensor`s
as latent variables, and the generative network. `elbo` at minimum needs
`p(x|Z)` and assumes that all `StochasticTensor`s upstream of `p(x|Z)` are
the variational distributions. Use `register_prior` to register `Distribution`
priors for each `StochasticTensor`. Alternatively, pass in
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
*  <b>`variational_with_prior`</b>: dict from `StochasticTensor` q(Z) to
    `Distribution` p(Z). If `None`, defaults to all `StochasticTensor`
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
    `StochasticTensor`s or if priors are not `Distribution`s.
*  <b>`TypeError`</b>: if form is not a valid ELBOForms constant.
*  <b>`ValueError`</b>: if `variational_with_prior` is None and there are no
    `StochasticTensor`s upstream of `log_likelihood`.
*  <b>`ValueError`</b>: if any variational does not have a prior passed or registered.

