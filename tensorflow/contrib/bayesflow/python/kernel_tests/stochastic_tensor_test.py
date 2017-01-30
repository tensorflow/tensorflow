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
from tensorflow.contrib import distributions as distributions_lib
from tensorflow.contrib.bayesflow.python.ops import stochastic_gradient_estimators
from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

distributions = distributions_lib
sge = stochastic_gradient_estimators
st = stochastic_tensor


class StochasticTensorTest(test.TestCase):

  def testConstructionAndValue(self):
    with self.test_session() as sess:
      mu = [0.0, 0.1, 0.2]
      sigma = constant_op.constant([1.1, 1.2, 1.3])
      sigma2 = constant_op.constant([0.1, 0.2, 0.3])

      prior_default = st.StochasticTensor(
          distributions.Normal(
              mu=mu, sigma=sigma))
      self.assertTrue(isinstance(prior_default.value_type, st.SampleValue))
      prior_0 = st.StochasticTensor(
          distributions.Normal(
              mu=mu, sigma=sigma),
          dist_value_type=st.SampleValue())
      self.assertTrue(isinstance(prior_0.value_type, st.SampleValue))

      with st.value_type(st.SampleValue()):
        prior = st.StochasticTensor(distributions.Normal(mu=mu, sigma=sigma))
        self.assertTrue(isinstance(prior.value_type, st.SampleValue))
        likelihood = st.StochasticTensor(
            distributions.Normal(
                mu=prior, sigma=sigma2))
        self.assertTrue(isinstance(likelihood.value_type, st.SampleValue))

      coll = ops.get_collection(st.STOCHASTIC_TENSOR_COLLECTION)
      self.assertEqual(coll, [prior_default, prior_0, prior, likelihood])

      # Also works: tf.convert_to_tensor(prior)
      prior_default = array_ops.identity(prior_default)
      prior_0 = array_ops.identity(prior_0)
      prior = array_ops.identity(prior)
      likelihood = array_ops.identity(likelihood)

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
      sigma = constant_op.constant([1.1, 1.2, 1.3])

      with st.value_type(st.MeanValue()):
        prior = st.StochasticTensor(distributions.Normal(mu=mu, sigma=sigma))
        self.assertTrue(isinstance(prior.value_type, st.MeanValue))

      prior_mean = prior.mean()
      prior_value = prior.value()

      prior_mean_val, prior_value_val = sess.run([prior_mean, prior_value])
      self.assertAllEqual(prior_mean_val, mu)
      self.assertAllEqual(prior_mean_val, prior_value_val)

  def testSampleValueScalar(self):
    with self.test_session() as sess:
      mu = [[0.0, -1.0, 1.0], [0.0, -1.0, 1.0]]
      sigma = constant_op.constant([[1.1, 1.2, 1.3], [1.1, 1.2, 1.3]])

      with st.value_type(st.SampleValue()):
        prior_single = st.StochasticTensor(
            distributions.Normal(
                mu=mu, sigma=sigma))

      prior_single_value = prior_single.value()
      self.assertEqual(prior_single_value.get_shape(), (2, 3))

      prior_single_value_val = sess.run([prior_single_value])[0]
      self.assertEqual(prior_single_value_val.shape, (2, 3))

      with st.value_type(st.SampleValue(1)):
        prior_single = st.StochasticTensor(
            distributions.Normal(
                mu=mu, sigma=sigma))
        self.assertTrue(isinstance(prior_single.value_type, st.SampleValue))

      prior_single_value = prior_single.value()
      self.assertEqual(prior_single_value.get_shape(), (1, 2, 3))

      prior_single_value_val = sess.run([prior_single_value])[0]
      self.assertEqual(prior_single_value_val.shape, (1, 2, 3))

      with st.value_type(st.SampleValue(2)):
        prior_double = st.StochasticTensor(
            distributions.Normal(
                mu=mu, sigma=sigma))

      prior_double_value = prior_double.value()
      self.assertEqual(prior_double_value.get_shape(), (2, 2, 3))

      prior_double_value_val = sess.run([prior_double_value])[0]
      self.assertEqual(prior_double_value_val.shape, (2, 2, 3))

  def testDistributionEntropy(self):
    with self.test_session() as sess:
      mu = [0.0, -1.0, 1.0]
      sigma = constant_op.constant([1.1, 1.2, 1.3])
      with st.value_type(st.MeanValue()):
        prior = st.StochasticTensor(distributions.Normal(mu=mu, sigma=sigma))
        entropy = prior.entropy()
        deep_entropy = prior.distribution.entropy()
        expected_deep_entropy = distributions.Normal(
            mu=mu, sigma=sigma).entropy()
        entropies = sess.run([entropy, deep_entropy, expected_deep_entropy])
        self.assertAllEqual(entropies[2], entropies[0])
        self.assertAllEqual(entropies[1], entropies[0])

  def testSurrogateLoss(self):
    with self.test_session():
      mu = [[3.0, -4.0, 5.0], [6.0, -7.0, 8.0]]
      sigma = constant_op.constant(1.0)

      # With default
      with st.value_type(st.MeanValue(stop_gradient=True)):
        dt = st.StochasticTensor(distributions.Normal(mu=mu, sigma=sigma))
      loss = dt.loss([constant_op.constant(2.0)])
      self.assertTrue(loss is not None)
      self.assertAllClose(
          dt.distribution.log_prob(mu).eval() * 2.0, loss.eval())

      # With passed-in loss_fn.
      dt = st.StochasticTensor(
          distributions.Normal(
              mu=mu, sigma=sigma),
          dist_value_type=st.MeanValue(stop_gradient=True),
          loss_fn=sge.get_score_function_with_constant_baseline(
              baseline=constant_op.constant(8.0)))
      loss = dt.loss([constant_op.constant(2.0)])
      self.assertTrue(loss is not None)
      self.assertAllClose((dt.distribution.log_prob(mu) * (2.0 - 8.0)).eval(),
                          loss.eval())


class ValueTypeTest(test.TestCase):

  def testValueType(self):
    type_mean = st.MeanValue()
    type_reshape = st.SampleValue()
    type_full = st.SampleValue()
    with st.value_type(type_mean):
      self.assertEqual(st.get_current_value_type(), type_mean)
      with st.value_type(type_reshape):
        self.assertEqual(st.get_current_value_type(), type_reshape)
      with st.value_type(type_full):
        self.assertEqual(st.get_current_value_type(), type_full)
      self.assertEqual(st.get_current_value_type(), type_mean)
    with self.assertRaisesRegexp(ValueError, "No value type currently set"):
      st.get_current_value_type()


class ObservedStochasticTensorTest(test.TestCase):

  def testConstructionAndValue(self):
    with self.test_session() as sess:
      mu = [0.0, 0.1, 0.2]
      sigma = constant_op.constant([1.1, 1.2, 1.3])
      obs = array_ops.zeros((2, 3))
      z = st.ObservedStochasticTensor(
          distributions.Normal(
              mu=mu, sigma=sigma), value=obs)
      [obs_val, z_val] = sess.run([obs, z.value()])
      self.assertAllEqual(obs_val, z_val)

      coll = ops.get_collection(st.STOCHASTIC_TENSOR_COLLECTION)
      self.assertEqual(coll, [z])

  def testConstructionWithUnknownShapes(self):
    mu = array_ops.placeholder(dtypes.float32)
    sigma = array_ops.placeholder(dtypes.float32)
    obs = array_ops.placeholder(dtypes.float32)
    z = st.ObservedStochasticTensor(
        distributions.Normal(
            mu=mu, sigma=sigma), value=obs)

    mu2 = array_ops.placeholder(dtypes.float32, shape=[None])
    sigma2 = array_ops.placeholder(dtypes.float32, shape=[None])
    obs2 = array_ops.placeholder(dtypes.float32, shape=[None, None])
    z2 = st.ObservedStochasticTensor(
        distributions.Normal(
            mu=mu2, sigma=sigma2), value=obs2)

    coll = ops.get_collection(st.STOCHASTIC_TENSOR_COLLECTION)
    self.assertEqual(coll, [z, z2])

  def testConstructionErrors(self):
    mu = [0., 0.]
    sigma = [1., 1.]
    self.assertRaises(
        ValueError,
        st.ObservedStochasticTensor,
        distributions.Normal(
            mu=mu, sigma=sigma),
        value=array_ops.zeros((3,)))
    self.assertRaises(
        ValueError,
        st.ObservedStochasticTensor,
        distributions.Normal(
            mu=mu, sigma=sigma),
        value=array_ops.zeros((3, 1)))
    self.assertRaises(
        ValueError,
        st.ObservedStochasticTensor,
        distributions.Normal(
            mu=mu, sigma=sigma),
        value=array_ops.zeros(
            (1, 2), dtype=dtypes.int32))


if __name__ == "__main__":
  test.main()
