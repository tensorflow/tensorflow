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
"""Tests that the TensorFlow parts of the LSTM example run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.examples import lstm

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.platform import test


class _SeedRunConfig(estimator_lib.RunConfig):

  @property
  def tf_random_seed(self):
    return 3


class LSTMExampleTest(test.TestCase):

  def test_periodicity_learned(self):
    (observed_times, observed_values,
     all_times, predicted_values) = lstm.train_and_predict(
         training_steps=100, estimator_config=_SeedRunConfig(),
         export_directory=self.get_temp_dir())
    self.assertAllEqual([100], observed_times.shape)
    self.assertAllEqual([100, 5], observed_values.shape)
    self.assertAllEqual([200], all_times.shape)
    self.assertAllEqual([200, 5], predicted_values.shape)
    self.assertGreater(
        predicted_values[100, 4]
        - predicted_values[115, 4],  # Amplitude of fifth component
        0.2)


if __name__ == "__main__":
  test.main()
