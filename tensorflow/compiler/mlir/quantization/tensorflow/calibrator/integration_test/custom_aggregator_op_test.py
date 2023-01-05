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

# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model as quantize_model_wrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class CustomAggregatorTest(test.TestCase):

  def setUp(self):
    super(CustomAggregatorTest, self).setUp()
    ops.disable_eager_execution()

  def testBypassAndMinMax(self):
    with self.test_session():
      quantize_model_wrapper.clear_calibrator()
      input_tensor = array_ops.constant([1.0, 2.0, 3.0, 4.0, 5.0],
                                        dtypes.float32)
      aggregator = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor, '1')
      self.assertAllEqual(self.evaluate(aggregator), [1.0, 2.0, 3.0, 4.0, 5.0])
      min_val = quantize_model_wrapper.get_min_from_calibrator('1')
      max_val = quantize_model_wrapper.get_max_from_calibrator('1')
      self.assertAllEqual((min_val, max_val), (1.0, 5.0))

  def testTwoIdentities(self):
    with self.test_session():
      quantize_model_wrapper.clear_calibrator()
      input_tensor1 = array_ops.constant([1.0, 2.0, 3.0, 4.0, 5.0],
                                         dtypes.float32)
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1, '2')
      self.assertAllEqual(self.evaluate(aggregator1), [1.0, 2.0, 3.0, 4.0, 5.0])
      input_tensor2 = array_ops.constant([-1.0, -2.0, -3.0, -4.0, -5.0],
                                         dtypes.float32)
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2, '3')
      self.assertAllEqual(
          self.evaluate(aggregator2), [-1.0, -2.0, -3.0, -4.0, -5.0])

      min_val = quantize_model_wrapper.get_min_from_calibrator('2')
      max_val = quantize_model_wrapper.get_max_from_calibrator('2')
      self.assertAllEqual((min_val, max_val), (1.0, 5.0))
      min_val = quantize_model_wrapper.get_min_from_calibrator('3')
      max_val = quantize_model_wrapper.get_max_from_calibrator('3')
      self.assertAllEqual((min_val, max_val), (-5.0, -1.0))

  def testClearData(self):
    with self.test_session():
      quantize_model_wrapper.clear_calibrator()
      input_tensor1 = array_ops.constant([1.0, 2.0, 3.0, 4.0, 5.0],
                                         dtypes.float32)
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1, '4')
      self.assertAllEqual(self.evaluate(aggregator1), [1.0, 2.0, 3.0, 4.0, 5.0])
      input_tensor2 = array_ops.constant([-1.0, -2.0, -3.0, -4.0, -5.0],
                                         dtypes.float32)
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2, '5')
      self.assertAllEqual(
          self.evaluate(aggregator2), [-1.0, -2.0, -3.0, -4.0, -5.0])

      min_val = quantize_model_wrapper.get_min_from_calibrator('4')
      max_val = quantize_model_wrapper.get_max_from_calibrator('4')
      self.assertAllEqual((min_val, max_val), (1.0, 5.0))
      min_val = quantize_model_wrapper.get_min_from_calibrator('5')
      max_val = quantize_model_wrapper.get_max_from_calibrator('5')
      self.assertAllEqual((min_val, max_val), (-5.0, -1.0))

      quantize_model_wrapper.clear_data_from_calibrator('4')
      with self.assertRaises(ValueError):
        quantize_model_wrapper.get_min_from_calibrator('4')
      min_val = quantize_model_wrapper.get_min_from_calibrator('5')
      max_val = quantize_model_wrapper.get_max_from_calibrator('5')
      self.assertAllEqual((min_val, max_val), (-5.0, -1.0))


if __name__ == '__main__':
  test.main()
