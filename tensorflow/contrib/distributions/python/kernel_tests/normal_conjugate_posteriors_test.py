# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib import distributions as distributions_lib
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

distributions = distributions_lib


class NormalTest(test.TestCase):

  def testNormalConjugateKnownSigmaPosterior(self):
    with session.Session():
      mu0 = constant_op.constant([3.0])
      sigma0 = constant_op.constant([math.sqrt(10.0)])
      sigma = constant_op.constant([math.sqrt(2.0)])
      x = constant_op.constant([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      s = math_ops.reduce_sum(x)
      n = array_ops.size(x)
      prior = distributions.Normal(loc=mu0, scale=sigma0)
      posterior = distributions.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(posterior, distributions.Normal))
      posterior_log_pdf = posterior.log_prob(x).eval()
      self.assertEqual(posterior_log_pdf.shape, (6,))

  def testNormalConjugateKnownSigmaPosteriorND(self):
    with session.Session():
      batch_size = 6
      mu0 = constant_op.constant([[3.0, -3.0]] * batch_size)
      sigma0 = constant_op.constant([[math.sqrt(10.0), math.sqrt(15.0)]] *
                                    batch_size)
      sigma = constant_op.constant([[math.sqrt(2.0)]] * batch_size)
      x = array_ops.transpose(
          constant_op.constant(
              [[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=dtypes.float32))
      s = math_ops.reduce_sum(x)
      n = array_ops.size(x)
      prior = distributions.Normal(loc=mu0, scale=sigma0)
      posterior = distributions.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(posterior, distributions.Normal))
      posterior_log_pdf = posterior.log_prob(x).eval()
      self.assertEqual(posterior_log_pdf.shape, (6, 2))

  def testNormalConjugateKnownSigmaNDPosteriorND(self):
    with session.Session():
      batch_size = 6
      mu0 = constant_op.constant([[3.0, -3.0]] * batch_size)
      sigma0 = constant_op.constant([[math.sqrt(10.0), math.sqrt(15.0)]] *
                                    batch_size)
      sigma = constant_op.constant([[math.sqrt(2.0), math.sqrt(4.0)]] *
                                   batch_size)
      x = constant_op.constant(
          [[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], [2.5, -2.5, -4.0, 0.0, 1.0, -2.0]],
          dtype=dtypes.float32)
      s = math_ops.reduce_sum(x, reduction_indices=[1])
      x = array_ops.transpose(x)  # Reshape to shape (6, 2)
      n = constant_op.constant([6] * 2)
      prior = distributions.Normal(loc=mu0, scale=sigma0)
      posterior = distributions.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(posterior, distributions.Normal))

      # Calculate log_pdf under the 2 models
      posterior_log_pdf = posterior.log_prob(x)
      self.assertEqual(posterior_log_pdf.get_shape(), (6, 2))
      self.assertEqual(posterior_log_pdf.eval().shape, (6, 2))

  def testNormalConjugateKnownSigmaPredictive(self):
    with session.Session():
      batch_size = 6
      mu0 = constant_op.constant([3.0] * batch_size)
      sigma0 = constant_op.constant([math.sqrt(10.0)] * batch_size)
      sigma = constant_op.constant([math.sqrt(2.0)] * batch_size)
      x = constant_op.constant([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      s = math_ops.reduce_sum(x)
      n = array_ops.size(x)
      prior = distributions.Normal(loc=mu0, scale=sigma0)
      predictive = distributions.normal_conjugates_known_scale_predictive(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertTrue(isinstance(predictive, distributions.Normal))
      predictive_log_pdf = predictive.log_prob(x).eval()
      self.assertEqual(predictive_log_pdf.shape, (6,))


if __name__ == "__main__":
  test.main()
