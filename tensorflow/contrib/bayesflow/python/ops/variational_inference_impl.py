# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Variational inference.

See the ${@python/contrib.bayesflow.variational_inference} guide.

@@elbo
@@elbo_with_log_joint
@@ELBOForms
@@register_prior
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.bayesflow.python.ops import stochastic_graph as sg
from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor as st
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

VI_PRIORS = "__vi_priors__"


def register_prior(variational, prior):
  """Associate a variational `StochasticTensor` with a `Distribution` prior.

  This is a helper function used in conjunction with `elbo` that allows users
  to specify the mapping between variational distributions and their priors
  without having to pass in `variational_with_prior` explicitly.

  Args:
    variational: `StochasticTensor` q(Z). Approximating distribution.
    prior: `Distribution` p(Z). Prior distribution.

  Returns:
    None

  Raises:
    ValueError: if variational is not a `StochasticTensor` or `prior` is not
      a `Distribution`.
  """
  if not isinstance(variational, st.StochasticTensor):
    raise TypeError("variational must be a StochasticTensor")
  if not isinstance(prior, distribution.Distribution):
    raise TypeError("prior must be a Distribution")
  ops.add_to_collection(VI_PRIORS, (variational, prior))


class _ELBOForm(object):
  pass


class ELBOForms(object):
  """Constants to control the `elbo` calculation.

  `analytic_kl` uses the analytic KL divergence between the
  variational distribution(s) and the prior(s).

  `analytic_entropy` uses the analytic entropy of the variational
  distribution(s).

  `sample` uses the sample KL or the sample entropy is the joint is provided.

  See `elbo` for what is used with `default`.
  """
  default, analytic_kl, analytic_entropy, sample = (_ELBOForm()
                                                    for _ in range(4))

  @staticmethod
  def check_form(form):
    if form not in {
        ELBOForms.default, ELBOForms.analytic_kl, ELBOForms.analytic_entropy,
        ELBOForms.sample
    }:
      raise TypeError("form must be an ELBOForms constant")


def elbo(log_likelihood,
         variational_with_prior=None,
         keep_batch_dim=True,
         form=None,
         name="ELBO"):
  r"""Evidence Lower BOund. `log p(x) >= ELBO`.

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

  Args:
    log_likelihood: `Tensor` log p(x|Z).
    variational_with_prior: dict from `StochasticTensor` q(Z) to
      `Distribution` p(Z). If `None`, defaults to all `StochasticTensor`
      objects upstream of `log_likelihood` with priors registered with
      `register_prior`.
    keep_batch_dim: bool. Whether to keep the batch dimension when summing
      entropy/KL term. When the sample is per data point, this should be True;
      otherwise (e.g. in a Bayesian NN), this should be False.
    form: ELBOForms constant. Controls how the ELBO is computed. Defaults to
      ELBOForms.default.
    name: name to prefix ops with.

  Returns:
    `Tensor` ELBO of the same type and shape as `log_likelihood`.

  Raises:
    TypeError: if variationals in `variational_with_prior` are not
      `StochasticTensor`s or if priors are not `Distribution`s.
    TypeError: if form is not a valid ELBOForms constant.
    ValueError: if `variational_with_prior` is None and there are no
      `StochasticTensor`s upstream of `log_likelihood`.
    ValueError: if any variational does not have a prior passed or registered.
  """
  if form is None:
    form = ELBOForms.default
  with ops.name_scope(name):
    model = ops.convert_to_tensor(log_likelihood)
    variational_with_prior = _find_variational_and_priors(
        model, variational_with_prior)
    return _elbo(form, log_likelihood, None, variational_with_prior,
                 keep_batch_dim)


def elbo_with_log_joint(log_joint,
                        variational=None,
                        keep_batch_dim=True,
                        form=None,
                        name="ELBO"):
  """Evidence Lower BOund. `log p(x) >= ELBO`.

  This method is for models that have computed `p(x,Z)` instead of `p(x|Z)`.
  See `elbo` for further details.

  Because only the joint is specified, analytic KL is not available.

  Args:
    log_joint: `Tensor` log p(x, Z).
    variational: list of `StochasticTensor` q(Z). If `None`, defaults to all
      `StochasticTensor` objects upstream of `log_joint`.
    keep_batch_dim: bool. Whether to keep the batch dimension when summing
      entropy term. When the sample is per data point, this should be True;
      otherwise (e.g. in a Bayesian NN), this should be False.
    form: ELBOForms constant. Controls how the ELBO is computed. Defaults to
      ELBOForms.default.
    name: name to prefix ops with.

  Returns:
    `Tensor` ELBO of the same type and shape as `log_joint`.

  Raises:
    TypeError: if variationals in `variational` are not `StochasticTensor`s.
    TypeError: if form is not a valid ELBOForms constant.
    ValueError: if `variational` is None and there are no `StochasticTensor`s
      upstream of `log_joint`.
    ValueError: if form is ELBOForms.analytic_kl.
  """
  if form is None:
    form = ELBOForms.default
  if form == ELBOForms.analytic_kl:
    raise ValueError("ELBOForms.analytic_kl is not available when using "
                     "elbo_with_log_joint. Use elbo or a different form.")

  with ops.name_scope(name):
    model = ops.convert_to_tensor(log_joint)

    variational_with_prior = None
    if variational is not None:
      variational_with_prior = dict(zip(variational, [None] * len(variational)))
    variational_with_prior = _find_variational_and_priors(
        model, variational_with_prior, require_prior=False)
    return _elbo(form, None, log_joint, variational_with_prior, keep_batch_dim)


def _elbo(form, log_likelihood, log_joint, variational_with_prior,
          keep_batch_dim):
  """Internal implementation of ELBO. Users should use `elbo`.

  Args:
    form: ELBOForms constant. Controls how the ELBO is computed.
    log_likelihood: `Tensor` log p(x|Z).
    log_joint: `Tensor` log p(x, Z).
    variational_with_prior: `dict<StochasticTensor, Distribution>`, varational
      distributions to prior distributions.
    keep_batch_dim: bool. Whether to keep the batch dimension when reducing
      the entropy/KL.

  Returns:
    ELBO `Tensor` with same shape and dtype as `log_likelihood`/`log_joint`.
  """
  ELBOForms.check_form(form)

  # Order of preference
  # 1. Analytic KL: log_likelihood - KL(q||p)
  # 2. Analytic entropy: log_likelihood + log p(Z) + H[q], or log_joint + H[q]
  # 3. Sample: log_likelihood - (log q(Z) - log p(Z)) =
  #            log_likelihood + log p(Z) - log q(Z), or log_joint - q(Z)

  def _reduce(val):
    if keep_batch_dim:
      return val
    else:
      return math_ops.reduce_sum(val)

  kl_terms = []
  entropy_terms = []
  prior_terms = []
  for q, z, p in [(qz.distribution, qz.value(), pz)
                  for qz, pz in variational_with_prior.items()]:
    # Analytic KL
    kl = None
    if log_joint is None and form in {ELBOForms.default, ELBOForms.analytic_kl}:
      try:
        kl = kullback_leibler.kl(q, p)
        logging.info("Using analytic KL between q:%s, p:%s", q, p)
      except NotImplementedError as e:
        if form == ELBOForms.analytic_kl:
          raise e
    if kl is not None:
      kl_terms.append(-1. * _reduce(kl))
      continue

    # Analytic entropy
    entropy = None
    if form in {ELBOForms.default, ELBOForms.analytic_entropy}:
      try:
        entropy = q.entropy()
        logging.info("Using analytic entropy for q:%s", q)
      except NotImplementedError as e:
        if form == ELBOForms.analytic_entropy:
          raise e
    if entropy is not None:
      entropy_terms.append(_reduce(entropy))
      if log_likelihood is not None:
        prior = p.log_prob(z)
        prior_terms.append(_reduce(prior))
      continue

    # Sample
    if form in {ELBOForms.default, ELBOForms.sample}:
      entropy = -q.log_prob(z)
      entropy_terms.append(_reduce(entropy))
      if log_likelihood is not None:
        prior = p.log_prob(z)
        prior_terms.append(_reduce(prior))

  first_term = log_joint if log_joint is not None else log_likelihood
  return sum([first_term] + kl_terms + entropy_terms + prior_terms)


def _find_variational_and_priors(model,
                                 variational_with_prior,
                                 require_prior=True):
  """Find upstream StochasticTensors and match with registered priors."""
  if variational_with_prior is None:
    # pylint: disable=protected-access
    upstreams = sg._upstream_stochastic_nodes([model])
    # pylint: enable=protected-access
    upstreams = list(upstreams[model])
    if not upstreams:
      raise ValueError("No upstream stochastic nodes found for tensor: %s",
                       model)
    prior_map = dict(ops.get_collection(VI_PRIORS))
    variational_with_prior = {}
    for q in upstreams:
      if require_prior and (q not in prior_map or prior_map[q] is None):
        raise ValueError("No prior specified for StochasticTensor: %s", q)
      variational_with_prior[q] = prior_map.get(q)

  if not all(
      [isinstance(q, st.StochasticTensor) for q in variational_with_prior]):
    raise TypeError("variationals must be StochasticTensors")
  if not all([
      p is None or isinstance(p, distribution.Distribution)
      for p in variational_with_prior.values()
  ]):
    raise TypeError("priors must be Distribution objects")

  return variational_with_prior
