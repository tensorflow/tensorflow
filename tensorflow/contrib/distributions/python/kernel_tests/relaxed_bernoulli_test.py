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
"""Tests for the RelaxedBernoulli distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.special
from tensorflow.contrib.distributions.python.ops import relaxed_bernoulli
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test


class RelaxedBernoulliTest(test.TestCase):

  def testP(self):
    """Tests that parameter P is set correctly. Note that dist.p != dist.pdf."""
    temperature = 1.0
    p = [0.1, 0.4]
    dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p)
    with self.cached_session():
      self.assertAllClose(p, dist.probs.eval())

  def testLogits(self):
    temperature = 2.0
    logits = [-42., 42.]
    dist = relaxed_bernoulli.RelaxedBernoulli(temperature, logits=logits)
    with self.cached_session():
      self.assertAllClose(logits, dist.logits.eval())

    with self.cached_session():
      self.assertAllClose(scipy.special.expit(logits), dist.probs.eval())

    p = [0.01, 0.99, 0.42]
    dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p)
    with self.cached_session():
      self.assertAllClose(scipy.special.logit(p), dist.logits.eval())

  def testInvalidP(self):
    temperature = 1.0
    invalid_ps = [1.01, 2.]
    for p in invalid_ps:
      with self.cached_session():
        with self.assertRaisesOpError("probs has components greater than 1"):
          dist = relaxed_bernoulli.RelaxedBernoulli(temperature,
                                                    probs=p,
                                                    validate_args=True)
          dist.probs.eval()

    invalid_ps = [-0.01, -3.]
    for p in invalid_ps:
      with self.cached_session():
        with self.assertRaisesOpError("Condition x >= 0"):
          dist = relaxed_bernoulli.RelaxedBernoulli(temperature,
                                                    probs=p,
                                                    validate_args=True)
          dist.probs.eval()

    valid_ps = [0.0, 0.5, 1.0]
    for p in valid_ps:
      with self.cached_session():
        dist = relaxed_bernoulli.RelaxedBernoulli(temperature,
                                                  probs=p)
        self.assertEqual(p, dist.probs.eval())

  def testShapes(self):
    with self.cached_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        temperature = 1.0
        p = np.random.random(batch_shape).astype(np.float32)
        dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p)
        self.assertAllEqual(batch_shape, dist.batch_shape.as_list())
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([], dist.event_shape.as_list())
        self.assertAllEqual([], dist.event_shape_tensor().eval())

  def testZeroTemperature(self):
    """If validate_args, raises InvalidArgumentError when temperature is 0."""
    temperature = constant_op.constant(0.0)
    p = constant_op.constant([0.1, 0.4])
    dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p,
                                              validate_args=True)
    with self.cached_session():
      sample = dist.sample()
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sample.eval()

  def testDtype(self):
    temperature = constant_op.constant(1.0, dtype=dtypes.float32)
    p = constant_op.constant([0.1, 0.4], dtype=dtypes.float32)
    dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p)
    self.assertEqual(dist.dtype, dtypes.float32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.probs.dtype, dist.prob([0.0]).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob([0.0]).dtype)

    temperature = constant_op.constant(1.0, dtype=dtypes.float64)
    p = constant_op.constant([0.1, 0.4], dtype=dtypes.float64)
    dist64 = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p)
    self.assertEqual(dist64.dtype, dtypes.float64)
    self.assertEqual(dist64.dtype, dist64.sample(5).dtype)

  def testLogProb(self):
    with self.cached_session():
      t = np.array(1.0, dtype=np.float64)
      p = np.array(0.1, dtype=np.float64)  # P(x=1)
      dist = relaxed_bernoulli.RelaxedBernoulli(t, probs=p)
      xs = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
      # analytical density from Maddison et al. 2016
      alpha = np.array(p/(1-p), dtype=np.float64)
      expected_log_pdf = (np.log(t) + np.log(alpha) +
                          (-t-1)*(np.log(xs)+np.log(1-xs)) -
                          2*np.log(alpha*np.power(xs, -t) + np.power(1-xs, -t)))
      log_pdf = dist.log_prob(xs).eval()
      self.assertAllClose(expected_log_pdf, log_pdf)

  def testBoundaryConditions(self):
    with self.cached_session():
      temperature = 1e-2
      dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=1.0)
      self.assertAllClose(np.nan, dist.log_prob(0.0).eval())
      self.assertAllClose([np.nan], [dist.log_prob(1.0).eval()])

  def testSampleN(self):
    """mean of quantized samples still approximates the Bernoulli mean."""
    with self.cached_session():
      temperature = 1e-2
      p = [0.2, 0.6, 0.5]
      dist = relaxed_bernoulli.RelaxedBernoulli(temperature, probs=p)
      n = 10000
      samples = dist.sample(n)
      self.assertEqual(samples.dtype, dtypes.float32)
      sample_values = samples.eval()
      self.assertTrue(np.all(sample_values >= 0))
      self.assertTrue(np.all(sample_values <= 1))

      frac_ones_like = np.sum(sample_values >= 0.5, axis=0)/n
      self.assertAllClose(p, frac_ones_like, atol=1e-2)


if __name__ == "__main__":
  test.main()
