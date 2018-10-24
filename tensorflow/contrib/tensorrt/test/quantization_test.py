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

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


def build_graph(input_name, input_dims, output_name,
                add_quantization_nodes=False, dtype=dtypes.float32):
  def quantize(x, r):
    if add_quantization_nodes:
      x = gen_array_ops.fake_quant_with_min_max_vars(x, -r, r)
    return x
  g = ops.Graph()
  with g.as_default():
    x = array_ops.placeholder(
        dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
    x = quantize(x, 10.0)
    x = x + 5
    x = quantize(x, 15.0)
    x = x - 5
    x = quantize(x, 10.0)
    x = x * 0.1
    x = quantize(x, 1.0)
    x = array_ops.identity(x, name=output_name)
  return g

class QuantizationMissingAllRangesTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment with no quantization ranges."""
    input_name = "input"
    input_dims = [100, 100]
    output_name = "output"
    g = build_graph(input_name, input_dims, output_name,
                    add_quantization_nodes=False)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 100)])

  def ShouldRunTest(self, run_params):
    return (run_params.precision_mode == "INT8" and
            not run_params.use_optimizer and
            not run_params.dynamic_engine)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    if run_params.use_calibration:
      return ["my_trt_op_0"]
    return []

class QuantizationWithRangesTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment with no quantization ranges."""
    input_name = "input"
    input_dims = [100, 100]
    output_name = "output"
    g = build_graph(input_name, input_dims, output_name,
                    add_quantization_nodes=True)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 100)])

  def ShouldRunTest(self, run_params):
    return (run_params.precision_mode == "INT8" and
            not run_params.use_optimizer)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

class NonQuantizedPrecisionsWithRangesTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment with no quantization ranges."""
    input_name = "input"
    input_dims = [100, 100]
    output_name = "output"
    g = build_graph(input_name, input_dims, output_name,
                    add_quantization_nodes=True)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 100)])

  def ShouldRunTest(self, run_params):
    return (run_params.precision_mode == "FP32" or
            run_params.precision_mode == "FP16")

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0", "my_trt_op_1", "my_trt_op_2"]
  
  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

if __name__ == "__main__":
  test.main()
