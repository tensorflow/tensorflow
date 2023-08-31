# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for CalibrationAlgorithm."""

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_algorithm
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stats_pb2
from tensorflow.python.platform import test

_CalibrationMethod = quant_opts_pb2.CalibrationOptions.CalibrationMethod


class CalibrationAlgorithmTest(test.TestCase):

  def testMinMax(self):
    with self.test_session():
      calib_opts = quant_opts_pb2.CalibrationOptions(
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
      )
      statistics = calib_stats_pb2.CalibrationStatistics()
      statistics.min_max_statistics.global_min = 1.0
      statistics.min_max_statistics.global_max = 5.0

      min_value, max_value = calibration_algorithm.get_min_max_value(
          statistics, calib_opts
      )
      self.assertAllEqual((min_value, max_value), (1.0, 5.0))

  def testAverageMinMax(self):
    with self.test_session():
      calib_opts = quant_opts_pb2.CalibrationOptions(
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX
      )
      statistics = calib_stats_pb2.CalibrationStatistics()
      statistics.average_min_max_statistics.min_sum = 5.0
      statistics.average_min_max_statistics.max_sum = 50.0
      statistics.average_min_max_statistics.num_samples = 5

      min_value, max_value = calibration_algorithm.get_min_max_value(
          statistics, calib_opts
      )
      self.assertAllEqual((min_value, max_value), (1.0, 10.0))


if __name__ == '__main__':
  test.main()
