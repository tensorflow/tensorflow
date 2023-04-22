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
"""Model script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _GraphFn(x, add_quantization_nodes):

  def _Quantize(x, r):
    if add_quantization_nodes:
      x = gen_array_ops.fake_quant_with_min_max_vars(x, -r, r)
    return x

  x = _Quantize(x, 10.0)
  x = x + 5
  x = _Quantize(x, 15.0)
  x = x - 5
  x = _Quantize(x, 10.0)
  x = x * 0.1
  x = _Quantize(x, 1.0)
  w = constant_op.constant(np.ones((8, 1)), dtype=dtypes.float32)
  x = math_ops.matmul(x, w)
  x = _Quantize(x, 10.0)
  return array_ops.identity(x, name="output_0")


def _GetParams(self):
  return self.BuildParams(self.GraphFn, dtypes.float32, [[8, 8]], [[8, 1]])


class QuantizationMissingAllRangesTest(trt_test.TfTrtIntegrationTestBase):
  """Create a graph containing single segment with no quantization ranges."""

  def GraphFn(self, x):
    return _GraphFn(x, add_quantization_nodes=False)

  def GetParams(self):
    return _GetParams(self)

  def ShouldRunTest(self, run_params):
    # Only test static engine mode, with or without calibration.
    return (trt_test.IsTensorRTVersionGreaterEqual(5) and
            trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.convert_online and not run_params.dynamic_engine
           ), "test static engine, offline conversion and INT8"

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    # In static engine mode with calibration, it should build a calibration
    # engine.
    # In static engine mode without calibration, the engine building will
    # succeed but fall back to non-quantized ops.
    return ["TRTEngineOp_0"]


class QuantizationWithRangesTest(trt_test.TfTrtIntegrationTestBase):
  """Create a graph containing single segment with no quantization ranges."""

  def GraphFn(self, x):
    return _GraphFn(x, add_quantization_nodes=True)

  def GetParams(self):
    return _GetParams(self)

  def ShouldRunTest(self, run_params):
    # Test static/dynamic engine with/without calibration.
    return (trt_test.IsTensorRTVersionGreaterEqual(5) and
            trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.convert_online), "test offline conversion and INT8"

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01


class NonQuantizedPrecisionsWithRangesTest(trt_test.TfTrtIntegrationTestBase):
  """Create a graph containing single segment with no quantization ranges."""

  def GraphFn(self, x):
    return _GraphFn(x, add_quantization_nodes=True)

  def GetParams(self):
    return _GetParams(self)

  def ShouldRunTest(self, run_params):
    # Only test FP32/FP16 mode.
    return not trt_test.IsQuantizationMode(
        run_params.precision_mode), "test non-INT8"

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    # The fake quant ops are not supported in FP32/FP16 mode, and will split the
    # graph into three TRT segments.
    return ["TRTEngineOp_0", "TRTEngineOp_1", "TRTEngineOp_2", "TRTEngineOp_3"]

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01


if __name__ == "__main__":
  test.main()
