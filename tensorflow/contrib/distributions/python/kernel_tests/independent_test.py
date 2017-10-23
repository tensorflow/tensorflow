# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the Independent distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import numpy as np

from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.contrib.distributions.python.ops import mvn_diag as mvn_diag_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module

stats = try_import("scipy.stats")


class ProductDistributionTest(test.TestCase):

  def testSampleAndLogProbUnivariate(self):
    loc = np.float32([-1., 1])
    scale = np.float32([0.1, 0.5])
    with self.test_session() as sess:
      ind = independent_lib.Independent(
          distribution=normal_lib.Normal(loc=loc, scale=scale),
          reduce_batch_ndims=1)

      x = ind.sample([4, 5])
      log_prob_x = ind.log_prob(x)
      x_, actual_log_prob_x = sess.run([x, log_prob_x])

      self.assertEqual([], ind.batch_shape)
      self.assertEqual([2], ind.event_shape)
      self.assertEqual([4, 5, 2], x.shape)
      self.assertEqual([4, 5], log_prob_x.shape)

      expected_log_prob_x = stats.norm(loc, scale).logpdf(x_).sum(-1)
      self.assertAllClose(expected_log_prob_x, actual_log_prob_x,
                          rtol=1e-5, atol=0.)

  def testSampleAndLogProbMultivariate(self):
    loc = np.float32([[-1., 1], [1, -1]])
    scale = np.float32([1., 0.5])
    with self.test_session() as sess:
      ind = independent_lib.Independent(
          distribution=mvn_diag_lib.MultivariateNormalDiag(
              loc=loc,
              scale_identity_multiplier=scale),
          reduce_batch_ndims=1)

      x = ind.sample([4, 5])
      log_prob_x = ind.log_prob(x)
      x_, actual_log_prob_x = sess.run([x, log_prob_x])

      self.assertEqual([], ind.batch_shape)
      self.assertEqual([2, 2], ind.event_shape)
      self.assertEqual([4, 5, 2, 2], x.shape)
      self.assertEqual([4, 5], log_prob_x.shape)

      expected_log_prob_x = stats.norm(loc, scale[:, None]).logpdf(
          x_).sum(-1).sum(-1)
      self.assertAllClose(expected_log_prob_x, actual_log_prob_x,
                          rtol=1e-6, atol=0.)

  def testSampleConsistentStats(self):
    loc = np.float32([[-1., 1], [1, -1]])
    scale = np.float32([1., 0.5])
    n_samp = 1e4
    with self.test_session() as sess:
      ind = independent_lib.Independent(
          distribution=mvn_diag_lib.MultivariateNormalDiag(
              loc=loc,
              scale_identity_multiplier=scale),
          reduce_batch_ndims=1)

      x = ind.sample(int(n_samp), seed=42)
      sample_mean = math_ops.reduce_mean(x, axis=0)
      sample_var = math_ops.reduce_mean(
          math_ops.squared_difference(x, sample_mean), axis=0)
      sample_std = math_ops.sqrt(sample_var)
      sample_entropy = -math_ops.reduce_mean(ind.log_prob(x), axis=0)

      [
          sample_mean_, sample_var_, sample_std_, sample_entropy_,
          actual_mean_, actual_var_, actual_std_, actual_entropy_,
          actual_mode_,
      ] = sess.run([
          sample_mean, sample_var, sample_std, sample_entropy,
          ind.mean(), ind.variance(), ind.stddev(), ind.entropy(), ind.mode(),
      ])

      self.assertAllClose(sample_mean_, actual_mean_, rtol=0.02, atol=0.)
      self.assertAllClose(sample_var_, actual_var_, rtol=0.04, atol=0.)
      self.assertAllClose(sample_std_, actual_std_, rtol=0.02, atol=0.)
      self.assertAllClose(sample_entropy_, actual_entropy_, rtol=0.01, atol=0.)
      self.assertAllClose(loc, actual_mode_, rtol=1e-6, atol=0.)


if __name__ == "__main__":
  test.main()
