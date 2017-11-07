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
"""Tests that the TensorFlow parts of the multivariate example run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.examples import multivariate

from tensorflow.python.platform import test


class MultivariateExampleTest(test.TestCase):

  def test_shapes_structural(self):
    times, values = multivariate.multivariate_train_and_sample(
        export_directory=self.get_temp_dir(), training_steps=5)
    self.assertAllEqual([1100], times.shape)
    self.assertAllEqual([1100, 5], values.shape)


if __name__ == "__main__":
  test.main()
