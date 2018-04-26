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
"""Tests that the TensorFlow parts of the known anomaly example run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.examples import known_anomaly

from tensorflow.python.platform import test


class KnownAnomalyExampleTest(test.TestCase):

  def test_shapes_and_variance_structural_ar(self):
    (times, observed, all_times, mean, upper_limit, lower_limit,
     anomaly_locations) = known_anomaly.train_and_evaluate_exogenous(
         train_steps=1, estimator_fn=known_anomaly.autoregressive_esitmator)
    self.assertAllEqual(
        anomaly_locations,
        [25, 50, 75, 100, 125, 150, 175, 249])
    self.assertAllEqual(all_times.shape, mean.shape)
    self.assertAllEqual(all_times.shape, upper_limit.shape)
    self.assertAllEqual(all_times.shape, lower_limit.shape)
    self.assertAllEqual(times.shape, observed.shape)

  def test_shapes_and_variance_structural_ssm(self):
    (times, observed, all_times, mean, upper_limit, lower_limit,
     anomaly_locations) = known_anomaly.train_and_evaluate_exogenous(
         train_steps=50, estimator_fn=known_anomaly.state_space_esitmator)
    self.assertAllEqual(
        anomaly_locations,
        [25, 50, 75, 100, 125, 150, 175, 249])
    self.assertAllEqual([200], times.shape)
    self.assertAllEqual([200], observed.shape)
    self.assertAllEqual([300], all_times.shape)
    self.assertAllEqual([300], mean.shape)
    self.assertAllEqual([300], upper_limit.shape)
    self.assertAllEqual([300], lower_limit.shape)
    # Check that initial predictions are relatively confident.
    self.assertLess(upper_limit[210] - lower_limit[210],
                    3.0 * (upper_limit[200] - lower_limit[200]))
    # Check that post-changepoint predictions are less confident
    self.assertGreater(upper_limit[290] - lower_limit[290],
                       3.0 * (upper_limit[240] - lower_limit[240]))

if __name__ == "__main__":
  test.main()
