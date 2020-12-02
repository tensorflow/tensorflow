# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test conversion of graphs involving INT32 tensors and operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CastFp32Fp16Test(trt_test.TfTrtIntegrationTestBase):
  """Tests cast to back and forth between FP16 and FP32."""

  def GraphFn(self, net):

    # Convert FP32 => FP16
    net = math_ops.cast(net, dtypes.float16, name="cast_fp32_to_fp16_1")
    net = math_ops.add(net, 1.0, name="add_fp16_1")
    net = math_ops.add(net, 1.0, name="add_fp16_2")

    # # Convert FP16 => FP32
    net = math_ops.cast(net, dtypes.float32, name="cast_fp16_to_fp32_1")
    net = math_ops.mul(net, 1.1, name="mul_fp32_1")
    net = math_ops.mul(net, 0.9, name="mul_fp32_2")

    # Convert FP32 => FP16
    net = math_ops.cast(net, dtypes.float16, name="cast_fp32_to_fp16_2")
    net = math_ops.mul(net, 1.1, name="mul_fp16_1")
    net = math_ops.mul(net, 0.9, name="mul_fp16_2")

    # Convert FP16 => FP16
    net = math_ops.cast(net, dtypes.float16, name="cast_fp16_to_fp16")

    # Convert FP16 => FP32
    net = math_ops.cast(net, dtypes.float32, name="cast_fp16_to_fp32_2")
    net = math_ops.add(net, 1.0, name="add_fp32_1")
    net = math_ops.add(net, 1.0, name="add_fp32_2")

    # Convert FP32 => FP32
    net = math_ops.cast(net, dtypes.float32, name="cast_fp32_to_fp32")

    return array_ops.identity(net, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 10]], [[1, 10]])

  def setUp(self):
    super(CastFp32Fp16Test, self).setUp()
    # Disable layout optimizer, since it will convert BiasAdd with NHWC
    # format to NCHW format under four dimensional input.
    self.DisableNonTrtOptimizers()

  def ExpectedAbsoluteTolerance(self, run_params):
      """The absolute tolerance to compare floating point results."""
      # We increase the tolerance due to the multiple casts back and forth
      return 5e-03

  def ExpectedRelativeTolerance(self, run_params):
      """The relative tolerance to compare floating point results."""
      # We increase the tolerance due to the multiple casts back and forth
      return 5e-03

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""

    if run_params.precision_mode != "FP32":
      return {
        "TRTEngineOp_0": [
          "cast_fp32_to_fp16_1", "add_fp16_1", "add_fp16_2",
          "cast_fp16_to_fp32_1", "mul_fp32_1", "mul_fp32_2",
          "cast_fp32_to_fp16_2", "mul_fp16_1", "mul_fp16_2",
          "cast_fp16_to_fp16",
          "cast_fp16_to_fp32_2", "add_fp32_1", "add_fp32_2",
          "cast_fp32_to_fp32"
        ]
      }
    else:
      return {
        "TRTEngineOp_0": ["add_fp16_1", "add_fp16_2"],
        "TRTEngineOp_3": ["mul_fp32_1", "mul_fp32_2"],
        "TRTEngineOp_2": ["mul_fp16_1", "mul_fp16_2"],
        "TRTEngineOp_1": ["add_fp32_1", "add_fp32_2", "cast_fp32_to_fp32"]
      }

  def ShouldRunTest(self, run_params):
    should_run, reason = super().ShouldRunTest(run_params)
    # Only run for TRT 7.0.0 and above.
    import os
    if os.environ.get('TFTRT_CAST_PYTEST_ALLOW_SEGFAULT', 0):
      # TODO DEKHTIARJonathan: Remove when fixed.
      return should_run and \
        trt_test.IsTensorRTVersionGreaterEqual(7), \
        reason + ' and TRT Version >= 7.0.0'
    else:
      return should_run and \
        not run_params.use_calibration and \
        trt_test.IsTensorRTVersionGreaterEqual(7), \
        reason + ' and TRT Version >= 7.0.0 and INT8 Calibration is not used.'


if __name__ == "__main__":
  test.main()
