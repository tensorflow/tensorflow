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

import numpy as np
import tensorflow as tf

sg = tf.contrib.bayesflow.stochastic_graph
distributions = tf.contrib.distributions


class NormalNotParam(distributions.Normal):

  @property
  def is_reparameterized(self):
    return False


class DistributionTensorTest(tf.test.TestCase):

  def testConstructionAndValue(self):
    with self.test_session() as sess:
      mu = [0.0, 0.1, 0.2]
      sigma = tf.constant([1.1, 1.2, 1.3])
      sigma2 = tf.constant([0.1, 0.2, 0.3])

      prior_default = sg.DistributionTensor(
          distributions.Normal, mu=mu, sigma=sigma)
      self.assertTrue(
          isinstance(prior_default.value_type, sg.SampleAndReshapeValue))
      prior_0 = sg.DistributionTensor(
          distributions.Normal, mu=mu, sigma=sigma,
          dist_value_type=sg.SampleAndReshapeValue())
      self.assertTrue(isinstance(prior_0.value_type, sg.SampleAndReshapeValue))

      with sg.value_type(sg.SampleAndReshapeValue()):
        prior = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
        self.assertTrue(isinstance(prior.value_type, sg.SampleAndReshapeValue))
        likelihood = sg.DistributionTensor(
            distributions.Normal, mu=prior, sigma=sigma2)
        self.assertTrue(
            isinstance(likelihood.value_type, sg.SampleAndReshapeValue))

      coll = tf.get_collection(sg.STOCHASTIC_TENSOR_COLLECTION)
      self.assertEqual(coll, [prior_default, prior_0, prior, likelihood])

      # Also works: tf.convert_to_tensor(prior)
      prior_default = tf.identity(prior_default)
      prior_0 = tf.identity(prior_0)
      prior = tf.identity(prior)
      likelihood = tf.identity(likelihood)

      # Mostly a smoke test for now...
      prior_0_val, prior_val, prior_default_val, _ = sess.run(
          [prior_0, prior, prior_default, likelihood])

      self.assertEqual(prior_0_val.shape, prior_val.shape)
      self.assertEqual(prior_default_val.shape, prior_val.shape)
      # These are different random samples from the same distribution,
      # so the values should differ.
      self.assertGreater(np.abs(prior_0_val - prior_val).sum(), 1e-6)
      self.assertGreater(np.abs(prior_default_val - prior_val).sum(), 1e-6)

  def testMeanValue(self):
    with self.test_session() as sess:
      mu = [0.0, -1.0, 1.0]
      sigma = tf.constant([1.1, 1.2, 1.3])

      with sg.value_type(sg.MeanValue()):
        prior = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
        self.assertTrue(isinstance(prior.value_type, sg.MeanValue))

      prior_mean = prior.mean()
      prior_value = prior.value()

      prior_mean_val, prior_value_val = sess.run([prior_mean, prior_value])
      self.assertAllEqual(prior_mean_val, mu)
      self.assertAllEqual(prior_mean_val, prior_value_val)

  def testSampleAndReshapeValue(self):
    with self.test_session() as sess:
      mu = [[0.0, -1.0, 1.0], [0.0, -1.0, 1.0]]
      sigma = tf.constant([[1.1, 1.2, 1.3], [1.1, 1.2, 1.3]])

      with sg.value_type(sg.SampleAndReshapeValue()):
        prior_single = sg.DistributionTensor(
            distributions.Normal, mu=mu, sigma=sigma)

      prior_single_value = prior_single.value()
      self.assertEqual(prior_single_value.get_shape(), (2, 3))

      prior_single_value_val = sess.run([prior_single_value])[0]
      self.assertEqual(prior_single_value_val.shape, (2, 3))

      with sg.value_type(sg.SampleAndReshapeValue(n=2)):
        prior_double = sg.DistributionTensor(
            distributions.Normal, mu=mu, sigma=sigma)

      prior_double_value = prior_double.value()
      self.assertEqual(prior_double_value.get_shape(), (4, 3))

      prior_double_value_val = sess.run([prior_double_value])[0]
      self.assertEqual(prior_double_value_val.shape, (4, 3))

  def testSampleValue(self):
    with self.test_session() as sess:
      mu = [[0.0, -1.0, 1.0], [0.0, -1.0, 1.0]]
      sigma = tf.constant([[1.1, 1.2, 1.3], [1.1, 1.2, 1.3]])

      with sg.value_type(sg.SampleValue()):
        prior_single = sg.DistributionTensor(
            distributions.Normal, mu=mu, sigma=sigma)
        self.assertTrue(isinstance(prior_single.value_type, sg.SampleValue))

      prior_single_value = prior_single.value()
      self.assertEqual(prior_single_value.get_shape(), (1, 2, 3))

      prior_single_value_val = sess.run([prior_single_value])[0]
      self.assertEqual(prior_single_value_val.shape, (1, 2, 3))

      with sg.value_type(sg.SampleValue(n=2)):
        prior_double = sg.DistributionTensor(
            distributions.Normal, mu=mu, sigma=sigma)

      prior_double_value = prior_double.value()
      self.assertEqual(prior_double_value.get_shape(), (2, 2, 3))

      prior_double_value_val = sess.run([prior_double_value])[0]
      self.assertEqual(prior_double_value_val.shape, (2, 2, 3))

  def testDistributionEntropy(self):
    with self.test_session() as sess:
      mu = [0.0, -1.0, 1.0]
      sigma = tf.constant([1.1, 1.2, 1.3])
      with sg.value_type(sg.MeanValue()):
        prior = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
        entropy = prior.entropy()
        deep_entropy = prior.entropy()
        expected_deep_entropy = distributions.Normal(
            mu=mu, sigma=sigma).entropy()
        entropies = sess.run([entropy, deep_entropy, expected_deep_entropy])
        self.assertAllEqual(entropies[2], entropies[0])
        self.assertAllEqual(entropies[1], entropies[0])

  def testSurrogateLoss(self):
    with self.test_session():
      mu = [[3.0, -4.0, 5.0], [6.0, -7.0, 8.0]]
      sigma = tf.constant(1.0)

      # With default
      with sg.value_type(sg.MeanValue(stop_gradient=True)):
        dt = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
      loss = dt.loss([tf.constant(2.0)])
      self.assertTrue(loss is not None)
      self.assertAllClose(dt.distribution.log_prob(mu).eval() * 2.0,
                          loss.eval())

      # With passed-in loss_fn.
      dt = sg.DistributionTensor(
          distributions.Normal,
          mu=mu,
          sigma=sigma,
          dist_value_type=sg.MeanValue(stop_gradient=True),
          loss_fn=sg.get_score_function_with_baseline(
              baseline=tf.constant(8.0)))
      loss = dt.loss([tf.constant(2.0)])
      self.assertTrue(loss is not None)
      self.assertAllClose((dt.distribution.log_prob(mu) * (2.0 - 8.0)).eval(),
                          loss.eval())


class ValueTypeTest(tf.test.TestCase):

  def testValueType(self):
    type_mean = sg.MeanValue()
    type_reshape = sg.SampleAndReshapeValue()
    type_full = sg.SampleValue()
    with sg.value_type(type_mean):
      self.assertEqual(sg.get_current_value_type(), type_mean)
      with sg.value_type(type_reshape):
        self.assertEqual(sg.get_current_value_type(), type_reshape)
      with sg.value_type(type_full):
        self.assertEqual(sg.get_current_value_type(), type_full)
      self.assertEqual(sg.get_current_value_type(), type_mean)
    with self.assertRaisesRegexp(ValueError, "No value type currently set"):
      sg.get_current_value_type()


class TestSurrogateLosses(tf.test.TestCase):

  def testPathwiseDerivativeDoesNotAddSurrogateLosses(self):
    with self.test_session():
      mu = [0.0, 0.1, 0.2]
      sigma = tf.constant([1.1, 1.2, 1.3])
      with sg.value_type(sg.SampleAndReshapeValue()):
        prior = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
        likelihood = sg.DistributionTensor(
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
      with sg.value_type(sg.SampleAndReshapeValue()):
        prior = sg.DistributionTensor(NormalNotParam, mu=mu, sigma=sigma)
        likelihood = sg.DistributionTensor(
            NormalNotParam, mu=prior, sigma=sigma)
        prior_2 = sg.DistributionTensor(NormalNotParam, mu=mu, sigma=sigma)

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
      with sg.value_type(sg.SampleAndReshapeValue()):
        dt = sg.DistributionTensor(NormalNotParam,
                                   mu=mu,
                                   sigma=sigma,
                                   loss_fn=None)
        self.assertEqual(None, dt.loss(tf.constant([2.0])))

  def testExplicitStochasticTensors(self):
    with self.test_session() as sess:
      mu = tf.constant([0.0, 0.1, 0.2])
      sigma = tf.constant([1.1, 1.2, 1.3])
      with sg.value_type(sg.SampleAndReshapeValue()):
        dt1 = sg.DistributionTensor(NormalNotParam, mu=mu, sigma=sigma)
        dt2 = sg.DistributionTensor(NormalNotParam, mu=mu, sigma=sigma)
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


if __name__ == "__main__":
  tf.test.main()
