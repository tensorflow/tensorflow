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
"""Tests for filtering postprocessors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries.state_space_models import filtering_postprocessor

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class FilteringStepPostprocessorTest(test.TestCase):

  def test_gaussian_alternative(self):
    for float_dtype in [dtypes.float32, dtypes.float64]:
      detector = filtering_postprocessor.StateInterpolatingAnomalyDetector(
          anomaly_log_likelihood=(filtering_postprocessor
                                  .cauchy_alternative_to_gaussian),
          responsibility_scaling=10.)
      predicted_state = [
          constant_op.constant(
              [[40.], [20.]], dtype=float_dtype), constant_op.constant(
                  [3., 6.], dtype=float_dtype), constant_op.constant([-1, -2])
      ]
      filtered_state = [
          constant_op.constant(
              [[80.], [180.]], dtype=float_dtype), constant_op.constant(
                  [1., 2.], dtype=float_dtype), constant_op.constant([-1, -2])
      ]
      interpolated_state, updated_outputs = detector.process_filtering_step(
          current_times=constant_op.constant([1, 2]),
          current_values=constant_op.constant([[0.], [1.]], dtype=float_dtype),
          predicted_state=predicted_state,
          filtered_state=filtered_state,
          outputs={
              "mean":
                  constant_op.constant([[0.1], [10.]], dtype=float_dtype),
              "covariance":
                  constant_op.constant([[[1.0]], [[1.0]]], dtype=float_dtype),
              "log_likelihood":
                  constant_op.constant([-1., -40.], dtype=float_dtype)
          })
      # The first batch element is not anomalous, and so should use the inferred
      # state. The second is anomalous, and should use the predicted state.
      expected_state = [[[80.], [20.]],
                        [1., 6.],
                        [-1, -2]]
      with self.cached_session():
        for interpolated, expected in zip(interpolated_state, expected_state):
          self.assertAllClose(expected, interpolated.eval())
        self.assertGreater(0., updated_outputs["anomaly_score"][0].eval())
        self.assertLess(0., updated_outputs["anomaly_score"][1].eval())

if __name__ == "__main__":
  test.main()
