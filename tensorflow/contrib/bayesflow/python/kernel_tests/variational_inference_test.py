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
"""Tests for variational inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib import distributions as distributions_lib
from tensorflow.contrib import layers
from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor
from tensorflow.contrib.bayesflow.python.ops import variational_inference
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.distributions.python.ops import normal
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

st = stochastic_tensor
vi = variational_inference
distributions = distributions_lib


class NormalNoEntropy(distributions.Normal):

  def entropy(self):
    raise NotImplementedError("entropy not implemented")


# For mini-VAE
def inference_net(x, latent_size):
  return layers.linear(x, latent_size)


def generative_net(z, data_size):
  return layers.linear(z, data_size)


def mini_vae():
  x = [[-6., 3., 6.], [-8., 4., 8.]]
  prior = distributions.Normal(loc=0., scale=1.)
  variational = st.StochasticTensor(
      distributions.Normal(
          loc=inference_net(x, 1), scale=1.))
  vi.register_prior(variational, prior)
  px = distributions.Normal(loc=generative_net(variational, 3), scale=1.)
  log_likelihood = math_ops.reduce_sum(px.log_prob(x), 1)
  log_likelihood = array_ops.expand_dims(log_likelihood, -1)
  return x, prior, variational, px, log_likelihood


class VariationalInferenceTest(test.TestCase):

  def testDefaultVariationalAndPrior(self):
    _, prior, variational, _, log_likelihood = mini_vae()
    elbo = vi.elbo(log_likelihood)
    expected_elbo = log_likelihood - kullback_leibler.kl(
        variational.distribution, prior)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(*sess.run([expected_elbo, elbo]))

  def testExplicitVariationalAndPrior(self):
    with self.test_session() as sess:
      _, _, variational, _, log_likelihood = mini_vae()
      prior = normal.Normal(loc=3., scale=2.)
      elbo = vi.elbo(
          log_likelihood, variational_with_prior={variational: prior})
      expected_elbo = log_likelihood - kullback_leibler.kl(
          variational.distribution, prior)
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(*sess.run([expected_elbo, elbo]))

  def testExplicitForms(self):
    _, prior, variational, _, log_likelihood = mini_vae()

    elbos = []
    forms = vi.ELBOForms
    for form in [
        forms.default, forms.analytic_kl, forms.sample, forms.analytic_entropy
    ]:
      elbo = vi.elbo(
          log_likelihood=log_likelihood,
          variational_with_prior={variational: prior},
          form=form)
      elbos.append(elbo)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      log_likelihood_shape = array_ops.shape(log_likelihood).eval()
      for elbo in elbos:
        elbo.eval()
        elbo_shape = array_ops.shape(elbo).eval()
        self.assertAllEqual(log_likelihood_shape, elbo_shape)
        self.assertEqual(elbo.dtype, log_likelihood.dtype)

  def testDefaultsSampleKLWithoutAnalyticKLOrEntropy(self):
    x = constant_op.constant([[-6., 3., 6.]])

    prior = distributions.Bernoulli(0.5)
    variational = st.StochasticTensor(
        NormalNoEntropy(
            loc=inference_net(x, 1), scale=1.))
    vi.register_prior(variational, prior)
    px = distributions.Normal(loc=generative_net(variational, 3), scale=1.)
    log_likelihood = math_ops.reduce_sum(px.log_prob(x), 1)

    # No analytic KL available between prior and variational distributions.
    with self.assertRaisesRegexp(NotImplementedError, "No KL"):
      distributions.kl(variational.distribution, prior)

    elbo = vi.elbo(
        variational_with_prior={variational: prior},
        log_likelihood=log_likelihood)
    expected_elbo = log_likelihood + prior.log_prob(
        variational) - variational.distribution.log_prob(variational)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(*sess.run([expected_elbo, elbo]))

  def testElboWithLogJoint(self):
    with self.test_session() as sess:
      _, prior, variational, _, log_likelihood = mini_vae()
      log_joint = log_likelihood + prior.log_prob(variational)
      elbo = vi.elbo_with_log_joint(log_joint)
      sess.run(variables.global_variables_initializer())
      elbo.eval()


if __name__ == "__main__":
  test.main()
