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
"""Tests for stochastic graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

st = tf.contrib.bayesflow.stochastic_tensor
sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


class NormalNotParam(distributions.Normal):

  @property
  def is_reparameterized(self):
    return False


class TestSurrogateLosses(tf.test.TestCase):

  def testPathwiseDerivativeDoesNotAddSurrogateLosses(self):
    with self.test_session():
      mu = [0.0, 0.1, 0.2]
      sigma = tf.constant([1.1, 1.2, 1.3])
      with st.value_type(st.SampleAndReshapeValue()):
        prior = st.StochasticTensor(distributions.Normal, mu=mu, sigma=sigma)
        likelihood = st.StochasticTensor(
            distributions.Normal, mu=prior, sigma=sigma)
        self.assertTrue(prior.distribution.is_reparameterized)
        self.assertTrue(likelihood.distribution.is_reparameterized)

      loss = tf.square(tf.identity(likelihood) - [0.0, 0.1, 0.2])
      sum_loss = tf.reduce_sum(loss)

      surrogate_loss = sg.surrogate_loss([loss])
      with self.assertRaisesRegexp(ValueError, "dimensionality 1 or greater"):
        _ = sg.surrogate_loss([sum_loss])
      surrogate_from_both = sg.surrogate_loss(
          [loss, sum_loss * tf.ones_like(loss)])

      # Pathwise derivative terms do not require add'l surrogate loss terms.
      with self.test_session() as sess:
        self.assertAllClose(*sess.run([loss, surrogate_loss]))
        self.assertAllClose(*sess.run([(loss + sum_loss), surrogate_from_both]))

  def _testSurrogateLoss(self, session, losses, expected_addl_terms, xs):
    surrogate_loss = sg.surrogate_loss(losses)
    expected_surrogate_loss = tf.add_n(losses + expected_addl_terms)
    self.assertAllClose(*session.run([surrogate_loss, expected_surrogate_loss]))

    # Test backprop
    expected_grads = tf.gradients(ys=expected_surrogate_loss, xs=xs)
    surrogate_grads = tf.gradients(ys=surrogate_loss, xs=xs)
    self.assertEqual(len(expected_grads), len(surrogate_grads))
    grad_values = session.run(expected_grads + surrogate_grads)
    n_grad = len(expected_grads)
    self.assertAllClose(grad_values[:n_grad], grad_values[n_grad:])

  def testSurrogateLoss(self):
    with self.test_session() as sess:
      mu = tf.constant([0.0, 0.1, 0.2])
      sigma = tf.constant([1.1, 1.2, 1.3])
      with st.value_type(st.SampleAndReshapeValue()):
        prior = st.StochasticTensor(NormalNotParam, mu=mu, sigma=sigma)
        likelihood = st.StochasticTensor(
            NormalNotParam, mu=prior, sigma=sigma)
        prior_2 = st.StochasticTensor(NormalNotParam, mu=mu, sigma=sigma)

      loss = tf.square(tf.identity(likelihood) - mu)
      part_loss = tf.square(tf.identity(prior) - mu)
      sum_loss = tf.reduce_sum(loss)
      loss_nodeps = tf.square(tf.identity(prior_2) - mu)

      # For ground truth, use the stop-gradient versions of the losses
      loss_nograd = tf.stop_gradient(loss)
      loss_nodeps_nograd = tf.stop_gradient(loss_nodeps)
      sum_loss_nograd = tf.stop_gradient(sum_loss)

      # These score functions should ignore prior_2
      self._testSurrogateLoss(
          session=sess,
          losses=[loss],
          expected_addl_terms=[
              likelihood.distribution.log_pdf(likelihood.value()) * loss_nograd,
              prior.distribution.log_pdf(prior.value()) * loss_nograd],
          xs=[mu, sigma])

      self._testSurrogateLoss(
          session=sess,
          losses=[loss, part_loss],
          expected_addl_terms=[
              likelihood.distribution.log_pdf(likelihood.value()) * loss_nograd,
              (prior.distribution.log_pdf(prior.value())
               * tf.stop_gradient(part_loss + loss))],
          xs=[mu, sigma])

      self._testSurrogateLoss(
          session=sess,
          losses=[sum_loss * tf.ones_like(loss)],
          expected_addl_terms=[
              (likelihood.distribution.log_pdf(likelihood.value())
               * sum_loss_nograd),
              prior.distribution.log_pdf(prior.value()) * sum_loss_nograd],
          xs=[mu, sigma])

      self._testSurrogateLoss(
          session=sess,
          losses=[loss, sum_loss * tf.ones_like(loss)],
          expected_addl_terms=[
              (likelihood.distribution.log_pdf(likelihood.value())
               * tf.stop_gradient(loss + sum_loss)),
              (prior.distribution.log_pdf(prior.value())
               * tf.stop_gradient(loss + sum_loss))],
          xs=[mu, sigma])

      # These score functions should ignore prior and likelihood
      self._testSurrogateLoss(
          session=sess,
          losses=[loss_nodeps],
          expected_addl_terms=[(prior_2.distribution.log_pdf(prior_2.value())
                                * loss_nodeps_nograd)],
          xs=[mu, sigma])

      # These score functions should include all terms selectively
      self._testSurrogateLoss(
          session=sess,
          losses=[loss, loss_nodeps],
          # We can't guarantee ordering of output losses in this case.
          expected_addl_terms=[
              (likelihood.distribution.log_pdf(likelihood.value())
               * loss_nograd),
              prior.distribution.log_pdf(prior.value()) * loss_nograd,
              (prior_2.distribution.log_pdf(prior_2.value())
               * loss_nodeps_nograd)],
          xs=[mu, sigma])

  def testNoSurrogateLoss(self):
    with self.test_session():
      mu = tf.constant([0.0, 0.1, 0.2])
      sigma = tf.constant([1.1, 1.2, 1.3])
      with st.value_type(st.SampleAndReshapeValue()):
        dt = st.StochasticTensor(NormalNotParam,
                                 mu=mu,
                                 sigma=sigma,
                                 loss_fn=None)
        self.assertEqual(None, dt.loss(tf.constant([2.0])))

  def testExplicitStochasticTensors(self):
    with self.test_session() as sess:
      mu = tf.constant([0.0, 0.1, 0.2])
      sigma = tf.constant([1.1, 1.2, 1.3])
      with st.value_type(st.SampleAndReshapeValue()):
        dt1 = st.StochasticTensor(NormalNotParam, mu=mu, sigma=sigma)
        dt2 = st.StochasticTensor(NormalNotParam, mu=mu, sigma=sigma)
        loss = tf.square(tf.identity(dt1)) + 10. + dt2

        sl_all = sg.surrogate_loss([loss])
        sl_dt1 = sg.surrogate_loss([loss], stochastic_tensors=[dt1])
        sl_dt2 = sg.surrogate_loss([loss], stochastic_tensors=[dt2])

        dt1_term = dt1.distribution.log_pdf(dt1) * loss
        dt2_term = dt2.distribution.log_pdf(dt2) * loss

        self.assertAllClose(*sess.run(
            [sl_all, sum([loss, dt1_term, dt2_term])]))
        self.assertAllClose(*sess.run([sl_dt1, sum([loss, dt1_term])]))
        self.assertAllClose(*sess.run([sl_dt2, sum([loss, dt2_term])]))


class StochasticDependenciesMapTest(tf.test.TestCase):

  def testBuildsMapOfUpstreamNodes(self):
    dt1 = st.StochasticTensor(distributions.Normal, mu=0., sigma=1.)
    dt2 = st.StochasticTensor(distributions.Normal, mu=0., sigma=1.)
    out1 = dt1.value() + 1.
    out2 = dt2.value() + 2.
    x = out1 + out2
    y = out2 * 3.
    dep_map = sg._stochastic_dependencies_map([x, y])
    self.assertEqual(dep_map[dt1], set([x]))
    self.assertEqual(dep_map[dt2], set([x, y]))

  def testHandlesStackedStochasticNodes(self):
    dt1 = st.StochasticTensor(distributions.Normal, mu=0., sigma=1.)
    out1 = dt1.value() + 1.
    dt2 = st.StochasticTensor(distributions.Normal, mu=out1, sigma=1.)
    x = dt2.value() + 2.
    dt3 = st.StochasticTensor(distributions.Normal, mu=0., sigma=1.)
    y = dt3.value() * 3.
    dep_map = sg._stochastic_dependencies_map([x, y])
    self.assertEqual(dep_map[dt1], set([x]))
    self.assertEqual(dep_map[dt2], set([x]))
    self.assertEqual(dep_map[dt3], set([y]))

  def testTraversesControlInputs(self):
    dt1 = st.StochasticTensor(distributions.Normal, mu=0., sigma=1.)
    logits = dt1.value() * 3.
    dt2 = st.StochasticTensor(distributions.Bernoulli, logits=logits)
    dt3 = st.StochasticTensor(distributions.Normal, mu=0., sigma=1.)
    x = dt3.value()
    y = tf.ones((2, 2)) * 4.
    z = tf.ones((2, 2)) * 3.
    out = tf.cond(
        tf.cast(dt2, tf.bool), lambda: tf.add(x, y), lambda: tf.square(z))
    out += 5.
    dep_map = sg._stochastic_dependencies_map([out])
    self.assertEqual(dep_map[dt1], set([out]))
    self.assertEqual(dep_map[dt2], set([out]))
    self.assertEqual(dep_map[dt3], set([out]))


if __name__ == "__main__":
  tf.test.main()
