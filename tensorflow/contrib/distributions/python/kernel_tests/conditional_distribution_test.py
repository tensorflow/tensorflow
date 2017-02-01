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
"""Tests for the Bernoulli distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import distributions
from tensorflow.contrib.distributions.python.kernel_tests import distribution_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class ConditionalDistributionTest(distribution_test.DistributionTest):

  def _GetFakeDistribution(self):
    class _FakeDistribution(distributions.ConditionalDistribution):
      """Fake Distribution for testing _set_sample_static_shape."""

      def __init__(self, batch_shape=None, event_shape=None):
        self._static_batch_shape = tensor_shape.TensorShape(batch_shape)
        self._static_event_shape = tensor_shape.TensorShape(event_shape)
        super(_FakeDistribution, self).__init__(
            dtype=dtypes.float32,
            is_continuous=False,
            reparameterization_type=distributions.NOT_REPARAMETERIZED,
            validate_args=True,
            allow_nan_stats=True,
            name="DummyDistribution")

      def _batch_shape(self):
        return self._static_batch_shape

      def _event_shape(self):
        return self._static_event_shape

      def _sample_n(self, unused_shape, unused_seed, arg1, arg2):
        raise ValueError(arg1, arg2)

      def _log_prob(self, _, arg1, arg2):
        raise ValueError(arg1, arg2)

      def _prob(self, _, arg1, arg2):
        raise ValueError(arg1, arg2)

      def _cdf(self, _, arg1, arg2):
        raise ValueError(arg1, arg2)

      def _log_cdf(self, _, arg1, arg2):
        raise ValueError(arg1, arg2)

      def _log_survival_function(self, _, arg1, arg2):
        raise ValueError(arg1, arg2)

      def _survival_function(self, _, arg1, arg2):
        raise ValueError(arg1, arg2)

    return _FakeDistribution

  def testNotImplemented(self):
    d = self._GetFakeDistribution()(batch_shape=[], event_shape=[])
    for name in ["sample", "log_prob", "prob", "log_cdf", "cdf",
                 "log_survival_function", "survival_function"]:
      method = getattr(d, name)
      with self.assertRaisesRegexp(ValueError, "b1.*b2"):
        method([] if name == "sample" else 1.0, arg1="b1", arg2="b2")


if __name__ == "__main__":
  test.main()
