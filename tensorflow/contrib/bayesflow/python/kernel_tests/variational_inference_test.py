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

import tensorflow as tf

st = tf.contrib.bayesflow.stochastic_tensor
vi = tf.contrib.bayesflow.variational_inference
distributions = tf.contrib.distributions


class NormalNoEntropy(distributions.Normal):

  def entropy(self):
    raise NotImplementedError("entropy not implemented")


# For mini-VAE
def inference_net(x, latent_size):
  return tf.contrib.layers.linear(x, latent_size)


def generative_net(z, data_size):
  return tf.contrib.layers.linear(z, data_size)


def mini_vae():
  x = [[-6., 3., 6.], [-8., 4., 8.]]
  prior = distributions.Normal(mu=0., sigma=1.)
  variational = st.StochasticTensor(
      distributions.Normal, mu=inference_net(x, 1), sigma=1.)
  vi.register_prior(variational, prior)
  px = distributions.Normal(mu=generative_net(variational, 3), sigma=1.)
  log_likelihood = tf.reduce_sum(px.log_prob(x), 1)
  log_likelihood = tf.expand_dims(log_likelihood, -1)
  return x, prior, variational, px, log_likelihood


class VariationalInferenceTest(tf.test.TestCase):

  def testDefaultVariationalAndPrior(self):
    _, prior, variational, _, log_likelihood = mini_vae()
    elbo = vi.elbo(log_likelihood)
    expected_elbo = log_likelihood - tf.contrib.distributions.kl(
        variational.distribution, prior)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      self.assertAllEqual(*sess.run([expected_elbo, elbo]))

  def testExplicitVariationalAndPrior(self):
    with self.test_session() as sess:
      _, _, variational, _, log_likelihood = mini_vae()
      prior = tf.contrib.distributions.Normal(mu=3., sigma=2.)
      elbo = vi.elbo(
          log_likelihood, variational_with_prior={variational: prior})
      expected_elbo = log_likelihood - tf.contrib.distributions.kl(
          variational.distribution, prior)
      sess.run(tf.initialize_all_variables())
      self.assertAllEqual(*sess.run([expected_elbo, elbo]))

  def testExplicitForms(self):
    _, prior, variational, _, log_likelihood = mini_vae()

    elbos = []
    forms = vi.ELBOForms
    for form in [forms.default, forms.analytic_kl, forms.sample,
                 forms.analytic_entropy]:
      elbo = vi.elbo(
          log_likelihood=log_likelihood,
          variational_with_prior={variational: prior},
          form=form)
      elbos.append(elbo)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      log_likelihood_shape = tf.shape(log_likelihood).eval()
      for elbo in elbos:
        elbo.eval()
        elbo_shape = tf.shape(elbo).eval()
        self.assertAllEqual(log_likelihood_shape, elbo_shape)
        self.assertEqual(elbo.dtype, log_likelihood.dtype)

  def testDefaultsSampleKLWithoutAnalyticKLOrEntropy(self):
    x = tf.constant([[-6., 3., 6.]])

    prior = distributions.Bernoulli(0.5)
    variational = st.StochasticTensor(
        NormalNoEntropy, mu=inference_net(x, 1), sigma=1.)
    vi.register_prior(variational, prior)
    px = distributions.Normal(mu=generative_net(variational, 3), sigma=1.)
    log_likelihood = tf.reduce_sum(px.log_prob(x), 1)

    # No analytic KL available between prior and variational distributions.
    with self.assertRaisesRegexp(NotImplementedError, "No KL"):
      distributions.kl(variational.distribution, prior)

    elbo = vi.elbo(
        variational_with_prior={variational: prior},
        log_likelihood=log_likelihood)
    expected_elbo = log_likelihood + prior.log_prob(
        variational) - variational.distribution.log_prob(variational)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      self.assertAllEqual(*sess.run([expected_elbo, elbo]))

  def testElboWithLogJoint(self):
    with self.test_session() as sess:
      _, prior, variational, _, log_likelihood = mini_vae()
      log_joint = log_likelihood + prior.log_prob(variational)
      elbo = vi.elbo_with_log_joint(log_joint)
      sess.run(tf.initialize_all_variables())
      elbo.eval()


if __name__ == "__main__":
  tf.test.main()
