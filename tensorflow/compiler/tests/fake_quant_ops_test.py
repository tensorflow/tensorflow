# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import googletest


class FakeQuantWithMinMaxArgsTest(xla_test.XLATestCase):
  """Test cases for FakeQuantWithMinMaxArgs operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    expected = np.array(
        [
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_max, expected_nudged_input_max,
            expected_nudged_input_max, expected_nudged_input_max
        ],
        dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        outputs = array_ops.fake_quant_with_min_max_args(
            input_placeholder,
            min=input_min,
            max=input_max,
            num_bits=num_bits,
            narrow_range=narrow_range)
      result = session.run(outputs, {input_placeholder: inputs})
      self.assertAllCloseAccordingToType(
          result, expected, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)


class FakeQuantWithMinMaxArgsGradientTest(xla_test.XLATestCase):
  """Test cases for FakeQuantWithMinMaxArgsGradient operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    gradients = np.arange(1, len(inputs) + 1, dtype=np.float32)
    expected_backprops = np.array(
        [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0],
        dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        gradient_placeholder = array_ops.placeholder(
            dtypes.float32, gradients.shape, name="gradients")
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        outputs = gen_array_ops.fake_quant_with_min_max_args_gradient(
            gradient_placeholder,
            input_placeholder,
            min=input_min,
            max=input_max,
            num_bits=num_bits,
            narrow_range=narrow_range)
      backprops = session.run(outputs, {
          gradient_placeholder: gradients,
          input_placeholder: inputs
      })
      self.assertAllCloseAccordingToType(
          backprops,
          expected_backprops,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)


class FakeQuantWithMinMaxVarsTest(xla_test.XLATestCase):
  """Test cases for FakeQuantWithMinMaxVars operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    expected = np.array(
        [
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_max, expected_nudged_input_max,
            expected_nudged_input_max, expected_nudged_input_max
        ],
        dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        min_placeholder = array_ops.placeholder(dtypes.float32, (), name="min")
        max_placeholder = array_ops.placeholder(dtypes.float32, (), name="max")
        outputs = array_ops.fake_quant_with_min_max_vars(
            input_placeholder,
            min_placeholder,
            max_placeholder,
            num_bits=num_bits,
            narrow_range=narrow_range)
      result = session.run(
          outputs, {
              input_placeholder: inputs,
              min_placeholder: input_min,
              max_placeholder: input_max
          })
      self.assertAllCloseAccordingToType(
          result, expected, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)


class FakeQuantWithMinMaxVarsGradientTest(xla_test.XLATestCase):
  """Test cases for FakeQuantWithMinMaxVarsGradient operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    gradients = np.arange(1, len(inputs) + 1, dtype=np.float32)
    expected_backprops_wrt_input = np.array(
        [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0],
        dtype=np.float32)
    expected_backprops_wrt_min = 1.0 + 2.0
    expected_backprops_wrt_max = 10.0 + 11.0

    with self.session() as session:
      with self.test_scope():
        gradient_placeholder = array_ops.placeholder(
            dtypes.float32, gradients.shape, name="gradients")
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        min_placeholder = array_ops.placeholder(dtypes.float32, (), name="min")
        max_placeholder = array_ops.placeholder(dtypes.float32, (), name="max")
        outputs = array_ops.fake_quant_with_min_max_vars_gradient(
            gradient_placeholder,
            input_placeholder,
            min_placeholder,
            max_placeholder,
            num_bits=num_bits,
            narrow_range=narrow_range)
      backprops_wrt_input, backprops_wrt_min, backprops_wrt_max = session.run(
          outputs, {
              gradient_placeholder: gradients,
              input_placeholder: inputs,
              min_placeholder: input_min,
              max_placeholder: input_max
          })
      self.assertAllCloseAccordingToType(
          backprops_wrt_input,
          expected_backprops_wrt_input,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)
      self.assertAllCloseAccordingToType(
          backprops_wrt_min,
          expected_backprops_wrt_min,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)
      self.assertAllCloseAccordingToType(
          backprops_wrt_max,
          expected_backprops_wrt_max,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)


if __name__ == "__main__":
  googletest.main()
