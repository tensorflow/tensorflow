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
"""Control Dependency tests for TF-TensorRT integration."""

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ControlDependencyTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing multiple segments."""
    c1 = constant_op.constant(1.0, name="c1")
    c2 = constant_op.constant(2.0, name="c2")
    d1 = self.trt_incompatible_op(inp, name="d1")
    d2 = self.trt_incompatible_binary_op(inp, inp, name="d2")
    with ops.control_dependencies([d1]):
      add = math_ops.add(inp, c1, name="add")
    mul = math_ops.mul(add, add, name="mul")
    add1 = math_ops.add(mul, mul, name="add1")
    edge = self.trt_incompatible_op(add1, name="incompatible")
    with ops.control_dependencies([d1, d2, add1]):
      add2 = math_ops.add(edge, c2, name="add2")
    mul1 = math_ops.mul(add2, add2, name="mul1")
    add3 = math_ops.add(mul1, mul1, name="add3")
    inc1 = self.trt_incompatible_binary_op(d1, add3, name="incompatible1")
    inc2 = self.trt_incompatible_binary_op(d2, inc1, name="incompatible2")
    return array_ops.squeeze(inc2, name="output_0")

  def GetParams(self):
    shapes = [[2, 32, 32, 3]]
    return self.BuildParams(self.GraphFn, dtypes.float32, input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["c1", "add", "add1", "mul"],
        "TRTEngineOp_001": ["c2", "add2", "add3", "mul1"]
    }

  def ExpectedConnections(self, run_params):
    """Returns the expected edges."""
    return {
        "input_0": set(),
        "d1": {"input_0"},
        "d2": {"input_0"},
        "TRTEngineOp_000": {"input_0", "^d1"},
        "incompatible": {"TRTEngineOp_000"},
        "TRTEngineOp_001": {"incompatible", "^d2"},
        "incompatible1": {"TRTEngineOp_001", "d1"},
        "incompatible2": {"incompatible1", "d2"},
        "output_0": {"incompatible2"},
    }

if __name__ == "__main__":
  test.main()
