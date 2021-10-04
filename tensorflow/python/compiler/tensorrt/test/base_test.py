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

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SimpleSingleEngineTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing single segment."""
    dtype = inp.dtype
    conv_filter = constant_op.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name="weights",
        dtype=dtype)
    conv = nn.conv2d(
        input=inp,
        filter=conv_filter,
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="conv")
    bias = constant_op.constant([4., 1.5, 2., 3., 5., 7.],
                                name="bias",
                                dtype=dtype)
    added = nn.bias_add(conv, bias, name="bias_add")
    relu = nn.relu(added, "relu")
    identity = array_ops.identity(relu, "identity")
    pool = nn_ops.max_pool(
        identity, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    return array_ops.squeeze(pool, name="output_0")

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]],
                            [[100, 6, 6, 6]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": [
            "weights", "conv", "bias", "bias_add", "relu", "identity",
            "max_pool"
        ]
    }


class SimpleMultiEnginesTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing multiple segment."""
    dtype = inp.dtype
    conv_filter = constant_op.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name="weights",
        dtype=dtype)
    conv = nn.conv2d(
        input=inp,
        filter=conv_filter,
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="conv")
    c1 = constant_op.constant(
        np.random.randn(12, 12, 6), dtype=dtype, name="c1")
    p = math_ops.mul(conv, c1, name="mul")
    c2 = constant_op.constant(
        np.random.randn(12, 12, 6), dtype=dtype, name="c2")
    q = math_ops.div(conv, c2, name="div")

    edge = self.trt_incompatible_op(q, name="incompatible")
    one = constant_op.constant(1, name="one", dtype=dtype)
    edge = math_ops.sub(one, edge, name="one_sub")
    edge = math_ops.div(edge, edge, name="div1")
    r = math_ops.add(edge, edge, name="add")

    p = math_ops.sub(p, edge, name="sub")
    q = math_ops.mul(q, edge, name="mul1")
    s = math_ops.add(p, q, name="add1")
    s = math_ops.sub(s, r, name="sub1")
    return array_ops.squeeze(s, name="output_0")

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]],
                            [[100, 12, 12, 6]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": [
            "add", "add1", "c1", "div1", "mul", "mul1", "sub", "sub1", "one",
            "one_sub"
        ],
        "TRTEngineOp_1": ["c2", "conv", "div", "weights"]
    }

  def setUp(self):
    super(trt_test.TfTrtIntegrationTestBase, self).setUp()  # pylint: disable=bad-super-call
    # Disable layout optimizer, since it will convert BiasAdd with NHWC
    # format to NCHW format under four dimentional input.
    self.DisableNonTrtOptimizers()


class SimpleMultiEnginesTest2(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing two segment."""
    n = inp
    for i in range(2):
      c = constant_op.constant(1.0, name="c%d" % i)
      n = math_ops.add(n, c, name="add%d" % i)
      n = math_ops.mul(n, n, name="mul%d" % i)
    edge = self.trt_incompatible_op(n, name="incompatible")
    with ops.control_dependencies([edge]):
      c = constant_op.constant(1.0, name="c2")
      n = math_ops.add(n, c, name="add2")
    n = math_ops.mul(n, n, name="mul2")
    c = constant_op.constant(1.0, name="c3")
    n = math_ops.add(n, c, name="add3")
    n = math_ops.mul(n, n, name="mul3")
    return array_ops.squeeze(n, name="output_0")

  def GetParams(self):
    shapes = [[2, 32, 32, 3]]
    return self.BuildParams(self.GraphFn, dtypes.float32, input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": ["c0", "c1", "add0", "add1", "mul0", "mul1"],
        "TRTEngineOp_1": ["c2", "c3", "add2", "add3", "mul2", "mul3"]
    }

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # Disable the test in fp16 mode since multiple matmul and add ops together
    # can cause overflow.
    return (
        (run_params.precision_mode != "FP16") and
        not (trt_test.IsQuantizationMode(run_params.precision_mode) and
             not run_params.use_calibration)), "test FP32 and non-calibration"


class ConstInputTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing multiple segment."""
    n = inp
    c = constant_op.constant(1.0, name="c")
    # Adds control dependency from the constant op to a trt incompatible op,
    # and adds control dependency from the trt incompatible op to all other
    # ops, to make sure the constant op cannot be contracted with any trt
    # segment that depends on it.
    with ops.control_dependencies([c]):
      d = self.trt_incompatible_op(n, name="incompatible")
    with ops.control_dependencies([d]):
      n = math_ops.add(n, c, name="add")
      n = math_ops.mul(n, n, name="mul")
      n = math_ops.add(n, n, name="add1")
    n = self.trt_incompatible_op(n, name="incompatible1")
    with ops.control_dependencies([d]):
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
        "TRTEngineOp_0": ["add", "add1", "mul"],
        "TRTEngineOp_1": ["add2", "add3", "mul1"]
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
    return {"TRTEngineOp_0": ["c", "add", "add1", "mul"]}


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
        "TRTEngineOp_0": ["add2", "add3", "mul1"],
        # Why segment ["add", "add1", "mul"] was assigned segment id 1
        # instead of 0: the parent node of this segment is actually const
        # node 'c', but it's removed later since it's const output of the
        # segment which is not allowed.
        "TRTEngineOp_1": ["add", "add1", "mul"]
    }


class ControlDependencyTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, inp):
    """Create a graph containing multiple segment."""
    c1 = constant_op.constant(1.0, name="c1")
    c2 = constant_op.constant(1.0, name="c2")
    d1 = constant_op.constant(1.0, name="d1")
    d2 = self.trt_incompatible_op(inp, name="d2")
    with ops.control_dependencies([d1, d2]):
      add = math_ops.add(inp, c1, name="add")
    with ops.control_dependencies([d1, d2]):
      mul = math_ops.mul(add, add, name="mul")
    with ops.control_dependencies([d1, d2]):
      add1 = math_ops.add(mul, mul, name="add1")
    edge = self.trt_incompatible_op(add1, name="incompatible")
    with ops.control_dependencies([d1, d2, add, mul]):
      add2 = math_ops.add(edge, c2, name="add2")
    with ops.control_dependencies([d1, d2, add1, mul]):
      mul1 = math_ops.mul(add2, add2, name="mul1")
    with ops.control_dependencies([d1, d2, add, add1]):
      add3 = math_ops.add(mul1, mul1, name="add3")
    return array_ops.squeeze(add3, name="output_0")

  def GetParams(self):
    shapes = [[2, 32, 32, 3]]
    return self.BuildParams(self.GraphFn, dtypes.float32, input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": ["c1", "add", "add1", "mul"],
        "TRTEngineOp_1": ["c2", "add2", "add3", "mul1"]
    }


if __name__ == "__main__":
  test.main()
