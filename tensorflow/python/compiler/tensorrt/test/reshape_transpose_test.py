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
"""Basic tests for TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ReshapeTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    outputs = []
    # Here we test two types of reshapes, one changes the batch dimension and
    # the other does not. Note that we're not able to test reshaping to
    # scalar, since TRT requires input tensor to be of rank at least 2, so a
    # reshape with scalar input will be filtered out of the segment before
    # conversion.
    #
    # These reshapes happen at batch dimension, thus conversion should fail.
    for shape in [[2, 50, 24, 24, 2], [-1, 50, 24, 24, 2], [2, 50, -1, 24, 2]]:
      incompatible_reshape = array_ops.reshape(inp, shape)
      reshape_back = array_ops.reshape(incompatible_reshape, [-1, 24, 24, 2])
      outputs.append(self.trt_incompatible_op(reshape_back))
    # Add another block with many reshapes that don't change the batch
    # dimension.
    compatible_reshape = array_ops.reshape(
        inp, [-1, 24 * 24, 2], name="reshape-0")
    compatible_reshape = array_ops.reshape(
        compatible_reshape, [100, 24, -1], name="reshape-1")
    compatible_reshape = array_ops.reshape(
        compatible_reshape, [100, 24 * 2, 24], name="reshape-2")
    compatible_reshape = array_ops.reshape(
        compatible_reshape, [-1, 24, 24 * 2], name="reshape-3")
    compatible_reshape = array_ops.reshape(
        compatible_reshape, [-1, 6, 4, 24, 2], name="reshape-4")
    compatible_reshape = array_ops.reshape(
        compatible_reshape, [-1, 6, 4, 6, 4, 2, 1], name="reshape-5")
    compatible_reshape = array_ops.reshape(
        compatible_reshape, [-1, 24, 24, 2], name="reshape-6")
    outputs.append(self.trt_incompatible_op(compatible_reshape))
    return math_ops.add_n(outputs, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]],
                            [[100, 24, 24, 2]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": ["reshape-%d" % i for i in range(7)] +
                         ["reshape-%d/shape" % i for i in range(7)]
    }

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.dynamic_engine), "test static engine and non-INT8"


class TransposeTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    # Add a block with compatible transposes.
    compatible_transpose = array_ops.transpose(
        inp, [0, 3, 1, 2], name="transpose-1")
    compatible_transpose = array_ops.transpose(
        compatible_transpose, [0, 2, 3, 1], name="transposeback")
    return array_ops.identity(compatible_transpose, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]],
                            [[100, 24, 24, 2]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": [
            "transpose-1", "transpose-1/perm", "transposeback",
            "transposeback/perm"
        ]
    }

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.dynamic_engine), "test static engine and non-INT8"


class IncompatibleTransposeTest(TransposeTest):

  def GraphFn(self, inp):
    # Add a block with incompatible transposes.
    incompatible_transpose = array_ops.transpose(
        inp, [2, 1, 0, 3], name="transpose-2")
    excluded_transpose = array_ops.transpose(
        incompatible_transpose, [0, 2, 3, 1], name="transpose-3")
    return array_ops.identity(excluded_transpose, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]],
                            [[24, 100, 2, 24]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []


if __name__ == "__main__":
  test.main()
