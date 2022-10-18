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
"""Constant inputs tests for TF-TensorRT integration."""

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ConstInputTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing multiple segment."""
    n = inp
    c = constant_op.constant(1.0, name="c")
    # Adds data dependency from the constant op to a trt incompatible op,
    # and adds data dependency from the trt incompatible op to the other
    # ops, to make sure the constant op cannot be contracted with any trt
    # segment that depends on it.
    n = self.trt_incompatible_binary_op(n, c, name="incompatible")
    n = math_ops.add(n, c, name="add")
    n = math_ops.mul(n, n, name="mul")
    n = math_ops.add(n, n, name="add1")
    n = self.trt_incompatible_op(n, name="incompatible1")
    n = math_ops.add(n, c, name="add2")
    n = math_ops.mul(n, n, name="mul1")
    n = math_ops.add(n, n, name="add3")
    return array_ops.squeeze(n, name="output_0")

  def GetParams(self):
    shapes = [[2, 32, 32, 3]]
    return self.BuildParams(self.GraphFn, dtypes.float32, input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["add", "add1", "mul"],
        "TRTEngineOp_001": ["add2", "add3", "mul1"]
    }

  def ExpectedConnections(self, run_params):
    """Returns the expected edges."""
    return {
        "input_0": set(),
        "c": set(),
        "incompatible": {"input_0", "c"},
        "TRTEngineOp_000": {"incompatible"},
        "incompatible1": {"TRTEngineOp_000"},
        "TRTEngineOp_001": {"incompatible1"},
        "output_0": {"TRTEngineOp_001"},
    }


class ConstDataInputSingleEngineTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing single segment."""
    n = inp
    c = constant_op.constant(1.0, name="c")
    n = math_ops.add(n, c, name="add")
    n = math_ops.mul(n, n, name="mul")
    n = math_ops.add(n, n, name="add1")
    return array_ops.squeeze(n, name="output_0")

  def GetParams(self):
    shapes = [[2, 32, 32, 3]]
    return self.BuildParams(self.GraphFn, dtypes.float32, input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["c", "add", "add1", "mul"]}


class ConstDataInputMultipleEnginesTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing multiple segment."""
    n = inp
    c = constant_op.constant(1.0, name="c")
    n = math_ops.add(n, c, name="add")
    n = math_ops.mul(n, n, name="mul")
    n = math_ops.add(n, n, name="add1")
    n = self.trt_incompatible_op(n, name="incompatible1")
    n = math_ops.add(n, c, name="add2")
    n = math_ops.mul(n, n, name="mul1")
    n = math_ops.add(n, n, name="add3")
    return array_ops.squeeze(n, name="output_0")

  def GetParams(self):
    shapes = [[2, 32, 32, 3]]
    return self.BuildParams(self.GraphFn, dtypes.float32, input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["add2", "add3", "mul1"],
        # Why segment ["add", "add1", "mul"] was assigned segment id 1
        # instead of 0: the parent node of this segment is actually const
        # node 'c', but it's removed later since it's const output of the
        # segment which is not allowed.
        "TRTEngineOp_001": ["add", "add1", "mul"]
    }


if __name__ == "__main__":
  test.main()
