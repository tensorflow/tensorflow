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
"""Tests for Custom Aggregator op."""

import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stat_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import pywrap_calibration
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)


class CustomAggregatorTest(test.TestCase):

  def setUp(self):
    super(CustomAggregatorTest, self).setUp()
    ops.disable_eager_execution()

  def testBypassAndMinMax(self):
    with self.session():
      pywrap_calibration.clear_calibrator()
      input_tensor = array_ops.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0], dtypes.float32
      )

      aggregator = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor,
          id='1',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator_output = self.evaluate(aggregator)
      self.assertAllEqual(aggregator_output.output, [1.0, 2.0, 3.0, 4.0, 5.0])
      self.assertEqual(aggregator_output.min, 1.0)
      self.assertEqual(aggregator_output.max, 5.0)
      self.assertEmpty(aggregator_output.histogram)

      statistics: calib_stat_pb2.CalibrationStatistics = (
          pywrap_calibration.get_statistics_from_calibrator('1')
      )

      min_val = statistics.min_max_statistics.global_min
      max_val = statistics.min_max_statistics.global_max

      self.assertAllEqual((min_val, max_val), (1.0, 5.0))

  def testTwoIdentities(self):
    with self.session():
      pywrap_calibration.clear_calibrator()
      input_tensor1 = array_ops.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0], dtypes.float32
      )
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1,
          '2',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator1_output = self.evaluate(aggregator1)
      self.assertAllEqual(aggregator1_output.output, [1.0, 2.0, 3.0, 4.0, 5.0])
      self.assertEqual(aggregator1_output.min, 1.0)
      self.assertEqual(aggregator1_output.max, 5.0)
      self.assertEmpty(aggregator1_output.histogram)

      input_tensor2 = array_ops.constant(
          [-1.0, -2.0, -3.0, -4.0, -5.0], dtypes.float32
      )
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2,
          '3',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator2_output = self.evaluate(aggregator2)
      self.assertAllEqual(
          aggregator2_output.output, [-1.0, -2.0, -3.0, -4.0, -5.0]
      )
      self.assertEqual(aggregator2_output.min, -5.0)
      self.assertEqual(aggregator2_output.max, -1.0)
      self.assertEmpty(aggregator2_output.histogram)

      statistics: calib_stat_pb2 = (
          pywrap_calibration.get_statistics_from_calibrator('2')
      )
      min_val = statistics.min_max_statistics.global_min
      max_val = statistics.min_max_statistics.global_max
      self.assertAllEqual((min_val, max_val), (1.0, 5.0))
      statistics: calib_stat_pb2 = (
          pywrap_calibration.get_statistics_from_calibrator('3')
      )
      min_val = statistics.min_max_statistics.global_min
      max_val = statistics.min_max_statistics.global_max
      self.assertAllEqual((min_val, max_val), (-5.0, -1.0))

  def testClearData(self):
    with self.session():
      pywrap_calibration.clear_calibrator()
      input_tensor1 = array_ops.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0], dtypes.float32
      )
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1,
          '4',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator1_output = self.evaluate(aggregator1)
      self.assertAllEqual(aggregator1_output.output, [1.0, 2.0, 3.0, 4.0, 5.0])
      self.assertEqual(aggregator1_output.min, 1.0)
      self.assertEqual(aggregator1_output.max, 5.0)
      self.assertEmpty(aggregator1_output.histogram)

      input_tensor2 = array_ops.constant(
          [-1.0, -2.0, -3.0, -4.0, -5.0], dtypes.float32
      )
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2,
          '5',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator2_output = self.evaluate(aggregator2)
      self.assertAllEqual(
          aggregator2_output.output, [-1.0, -2.0, -3.0, -4.0, -5.0]
      )
      self.assertEqual(aggregator2_output.min, -5.0)
      self.assertEqual(aggregator2_output.max, -1.0)
      self.assertEmpty(aggregator2_output.histogram)

      statistics: calib_stat_pb2 = (
          pywrap_calibration.get_statistics_from_calibrator('4')
      )
      min_val = statistics.min_max_statistics.global_min
      max_val = statistics.min_max_statistics.global_max
      self.assertAllEqual((min_val, max_val), (1.0, 5.0))

      statistics: calib_stat_pb2 = (
          pywrap_calibration.get_statistics_from_calibrator('5')
      )
      min_val = statistics.min_max_statistics.global_min
      max_val = statistics.min_max_statistics.global_max
      self.assertAllEqual((min_val, max_val), (-5.0, -1.0))

      pywrap_calibration.clear_data_from_calibrator('4')
      with self.assertRaises(ValueError):
        pywrap_calibration.get_statistics_from_calibrator('4')

      statistics: calib_stat_pb2 = (
          pywrap_calibration.get_statistics_from_calibrator('5')
      )
      min_val = statistics.min_max_statistics.global_min
      max_val = statistics.min_max_statistics.global_max
      self.assertAllEqual((min_val, max_val), (-5.0, -1.0))

  def testBypassAndAverageMinMax(self):
    with self.session():
      pywrap_calibration.clear_calibrator()
      input_tensor1 = array_ops.constant(
          [-50.0, -25.0, 0.0, 25.0, 50.0], dtypes.float32
      )
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1,
          '6',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX,
      )
      aggregator1_output = self.evaluate(aggregator1)
      self.assertAllEqual(
          aggregator1_output.output,
          [-50.0, -25.0, 0.0, 25.0, 50.0],
      )
      self.assertEqual(aggregator1_output.min, -50.0)
      self.assertEqual(aggregator1_output.max, 50.0)
      self.assertEmpty(aggregator1_output.histogram)

      input_tensor2 = array_ops.constant(
          [-100.0, -50.0, 0.0, 50.0, 100.0], dtypes.float32
      )
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2,
          '6',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX,
      )
      aggregator2_output = self.evaluate(aggregator2)
      self.assertAllEqual(
          aggregator2_output.output, [-100.0, -50.0, 0.0, 50.0, 100.0]
      )
      self.assertEqual(aggregator2_output.min, -100.0)
      self.assertEqual(aggregator2_output.max, 100.0)
      self.assertEmpty(aggregator2_output.histogram)

      statistics: calib_stat_pb2 = (
          pywrap_calibration.get_statistics_from_calibrator('6')
      )

      min_sum = statistics.average_min_max_statistics.min_sum
      max_sum = statistics.average_min_max_statistics.max_sum
      num_samples = statistics.average_min_max_statistics.num_samples

      self.assertAllEqual((min_sum, max_sum, num_samples), (-150.0, 150.0, 2))

  def testHistogramCalibration(self):
    with self.session():
      pywrap_calibration.clear_calibrator()
      input_tensor = array_ops.constant(
          [1.0, 1.0, 3.0, 4.0, 6.0], dtypes.float32
      )

      aggregator = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor,
          id='7',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
          initial_num_bins=256,
      )
      aggregator_output = self.evaluate(aggregator)
      self.assertAllEqual(aggregator_output.output, [1.0, 1.0, 3.0, 4.0, 6.0])
      self.assertEqual(aggregator_output.min, 1.0)
      self.assertEqual(aggregator_output.max, 6.0)

      self.assertLen(aggregator_output.histogram, 512)
      self.assertEqual(sum(aggregator_output.histogram), 5)
      self.assertEqual(aggregator_output.histogram[0], 2)
      self.assertEqual(aggregator_output.histogram[128], 1)
      self.assertEqual(aggregator_output.histogram[192], 1)
      self.assertEqual(aggregator_output.histogram[320], 1)


if __name__ == '__main__':
  test.main()
