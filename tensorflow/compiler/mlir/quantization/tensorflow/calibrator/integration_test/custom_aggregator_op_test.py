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
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
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

  def testTwoIdentities(self):
    with self.session():
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

  def testBypassAndAverageMinMax(self):
    with self.session():
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

  def testHistogramCalibration(self):
    with self.session():
      input_tensor = array_ops.constant(
          [1.0, 1.0, 3.0, 4.0, 6.0], dtypes.float32
      )

      aggregator = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor,
          id='7',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
          num_bins=512,
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
