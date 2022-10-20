# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for monitoring's Python bindings."""

from absl.testing import absltest
from tensorflow.tsl.lib.monitoring.python import monitoring


class MonitoringTest(absltest.TestCase):

  def test_new_sampler_metric_with_exponential_buckets(self):
    metric = monitoring.new_sampler_metric_with_exponential_buckets(
        '/test/metric',
        'Test if metric python bindings record values correctly.',
        # Exponential buckets:  Power of 2 with bucket count 20
        1000, 2, 20)
    metric.add(1.0)
    metric.add(100.0)
    metric.add(1000.0)
    self.assertEqual(metric.num_values(), 3)
    self.assertAlmostEqual(metric.sum(), 1101.0)


if __name__ == '__main__':
  absltest.main()
