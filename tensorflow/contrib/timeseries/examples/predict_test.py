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
"""Tests that the TensorFlow parts of the prediction example run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

from tensorflow.contrib.timeseries.examples import predict

from tensorflow.python.platform import test


_MODULE_PATH = path.dirname(__file__)
_DATA_FILE = path.join(_MODULE_PATH, "data/period_trend.csv")


class PeriodTrendExampleTest(test.TestCase):

  def test_shapes_and_variance_structural(self):
    (times, observed, all_times, mean, upper_limit, lower_limit
    ) = predict.structural_ensemble_train_and_predict(_DATA_FILE)
    # Just check that plotting will probably be OK. We can't actually run the
    # plotting code since we don't want to pull in matplotlib as a dependency
    # for this test.
    self.assertAllEqual([500], times.shape)
    self.assertAllEqual([500], observed.shape)
    self.assertAllEqual([700], all_times.shape)
    self.assertAllEqual([700], mean.shape)
    self.assertAllEqual([700], upper_limit.shape)
    self.assertAllEqual([700], lower_limit.shape)
    # Check that variance hasn't blown up too much. This is a relatively good
    # indication that training was successful.
    self.assertLess(upper_limit[-1] - lower_limit[-1],
                    1.5 * (upper_limit[0] - lower_limit[0]))

  def test_ar(self):
    (times, observed, all_times, mean,
     upper_limit, lower_limit) = predict.ar_train_and_predict(_DATA_FILE)
    self.assertAllEqual(times.shape, observed.shape)
    self.assertAllEqual(all_times.shape, mean.shape)
    self.assertAllEqual(all_times.shape, upper_limit.shape)
    self.assertAllEqual(all_times.shape, lower_limit.shape)
    self.assertLess((upper_limit - lower_limit).mean(), 4.)


if __name__ == "__main__":
  test.main()
