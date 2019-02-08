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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import autoregressive as autoregressive_lib
from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.contrib.distributions.python.ops.bijectors.affine import Affine
from tensorflow.contrib.distributions.python.ops.bijectors.masked_autoregressive import MaskedAutoregressiveFlow
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.platform import test


class AutogressiveTest(test_util.VectorDistributionTestHelpers, test.TestCase):
  """Tests the Autoregressive distribution."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_scale_tril(self, event_size):
    n = np.int32(event_size * (event_size + 1) // 2)
    p = 2. * self._rng.random_sample(n).astype(np.float32) - 1.
    return distribution_util.fill_triangular(0.25 * p)

  def _normal_fn(self, affine_bijector):
    def _fn(samples):
      scale = math_ops.exp(affine_bijector.forward(samples))
      return independent_lib.Independent(
          normal_lib.Normal(loc=0., scale=scale, validate_args=True),
          reinterpreted_batch_ndims=1)
    return _fn

  def testSampleAndLogProbConsistency(self):
    batch_shape = []
    event_size = 2
    with self.cached_session() as sess:
      batch_event_shape = np.concatenate([batch_shape, [event_size]], axis=0)
      sample0 = array_ops.zeros(batch_event_shape)
      affine = Affine(scale_tril=self._random_scale_tril(event_size))
      ar = autoregressive_lib.Autoregressive(
          self._normal_fn(affine), sample0, validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess.run, ar, radius=1., center=0., rtol=0.01)

  def testCompareToBijector(self):
    """Demonstrates equivalence between TD, Bijector approach and AR dist."""
    sample_shape = np.int32([4, 5])
    batch_shape = np.int32([])
    event_size = np.int32(2)
    with self.cached_session() as sess:
      batch_event_shape = np.concatenate([batch_shape, [event_size]], axis=0)
      sample0 = array_ops.zeros(batch_event_shape)
      affine = Affine(scale_tril=self._random_scale_tril(event_size))
      ar = autoregressive_lib.Autoregressive(
          self._normal_fn(affine), sample0, validate_args=True)
      ar_flow = MaskedAutoregressiveFlow(
          is_constant_jacobian=True,
          shift_and_log_scale_fn=lambda x: [None, affine.forward(x)],
          validate_args=True)
      td = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=ar_flow,
          event_shape=[event_size],
          batch_shape=batch_shape,
          validate_args=True)
      x_shape = np.concatenate(
          [sample_shape, batch_shape, [event_size]], axis=0)
      x = 2. * self._rng.random_sample(x_shape).astype(np.float32) - 1.
      td_log_prob_, ar_log_prob_ = sess.run([td.log_prob(x), ar.log_prob(x)])
      self.assertAllClose(td_log_prob_, ar_log_prob_, atol=0., rtol=1e-6)


if __name__ == "__main__":
  test.main()
