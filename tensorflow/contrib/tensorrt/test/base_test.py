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

import numpy as np

from tensorflow.contrib.tensorrt.python import trt_convert
from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SimpleSingleEngineTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    # TODO(aaroey): test graph with different dtypes.
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
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
        bias = constant_op.constant(
            [4., 1.5, 2., 3., 5., 7.], name="bias", dtype=dtype)
        added = nn.bias_add(conv, bias, name="bias_add")
        relu = nn.relu(added, "relu")
        identity = array_ops.identity(relu, "identity")
        pool = nn_ops.max_pool(
            identity, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
      array_ops.squeeze(pool, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 6, 6, 6)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    # TODO(aaroey): LayoutOptimizer adds additional nodes to the graph which
    # breaks the connection check, fix it.
    # - my_trt_op_0 should have ["weights", "conv", "bias", "bias_add",
    #   "relu", "identity", "max_pool"]
    return ["my_trt_op_0"]


class SimpleMultiEnginesTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing multiple segment."""
    # TODO(aaroey): test graph with different dtypes.
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
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
            np.random.randn(input_dims[0], 12, 12, 6), dtype=dtype, name="c1")
        p = math_ops.mul(conv, c1, name="mul")
        c2 = constant_op.constant(
            np.random.randn(input_dims[0], 12, 12, 6), dtype=dtype, name="c2")
        q = math_ops.div(conv, c2, name="div")

        edge = self.trt_incompatible_op(q, name="incompatible")
        edge = math_ops.div(edge, edge, name="div1")
        r = math_ops.add(edge, edge, name="add")

        p = math_ops.sub(p, edge, name="sub")
        q = math_ops.mul(q, edge, name="mul1")
        s = math_ops.add(p, q, name="add1")
        s = math_ops.sub(s, r, name="sub1")
      array_ops.squeeze(s, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 12, 12, 6)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    # TODO(aaroey): LayoutOptimizer adds additional nodes to the graph which
    # breaks the connection check, fix it.
    # - my_trt_op_0 should have ["mul", "sub", "div1", "mul1", "add1",
    #   "add", "sub1"];
    # - my_trt_op_1 should have ["weights","conv", "div"]
    return ["my_trt_op_0", "my_trt_op_1"]

  def ShouldRunTest(self, run_params):
    # TODO(aaroey): LayoutOptimizer adds Transpose(Const, Const) to the graph
    # which breaks the conversion. We should fix it as:
    # - Detect the invalid NodeDef earlier before adding them to segment
    # - Let it able to change the RewriterConfig when calling
    #   create_inference_graph().
    # It will be good to add debugging feature for Grappler to print the graph
    # after running each optimizer.
    return False


class PartiallyConvertedTestA(trt_test.TfTrtIntegrationTestBase):

  def setUp(self):
    """Setup method."""
    super(PartiallyConvertedTestA, self).setUp()
    # Let it fail to build the second engine.
    trt_convert.add_test_value("my_trt_op_1:CreateTRTNode", "fail")

  def GetParams(self):
    """Create a graph containing two segment."""
    input_name = "input"
    input_dims = [2, 32, 32, 3]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)
      with g.device("/GPU:0"):
        n = inp
        for i in range(2):
          c = constant_op.constant(1.0, name="c%d" % i)
          n = math_ops.add(n, c, name="add%d" % i)
          n = math_ops.mul(n, n, name="mul%d" % i)
        edge = self.trt_incompatible_op(n, name="incompatible")
        with g.control_dependencies([edge]):
          c = constant_op.constant(1.0, name="c2")
          n = math_ops.add(n, c, name="add2")
        n = math_ops.mul(n, n, name="mul2")
        c = constant_op.constant(1.0, name="c3")
        n = math_ops.add(n, c, name="add3")
        n = math_ops.mul(n, n, name="mul3")
      array_ops.squeeze(n, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[tuple(input_dims)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        # Only the first engine is built.
        "my_trt_op_0": ["c0", "c1", "add0", "add1", "mul0", "mul1"]
    }

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # Disable the test in fp16 mode since multiple matmul and add ops together
    # can cause overflow.
    return run_params.precision_mode != "FP16"


class PartiallyConvertedTestB(PartiallyConvertedTestA):

  def setUp(self):
    """Setup method."""
    super(PartiallyConvertedTestB, self).setUp()
    # Let it fail to build the first engine.
    trt_convert.clear_test_values("")
    trt_convert.add_test_value("my_trt_op_0:CreateTRTNode", "fail")

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        # Only the second engine is built.
        "my_trt_op_1": ["c2", "c3", "add2", "add3", "mul2", "mul3"]
    }


class ConstInputTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing multiple segment."""
    input_name = "input"
    input_dims = [2, 32, 32, 3]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)
      with g.device("/GPU:0"):
        n = inp
        c = constant_op.constant(1.0, name="c")
        # Adds control dependency from the constant op to a trt incompatible op,
        # and adds control dependency from the trt incompatible op to all other
        # ops, to make sure the constant op cannot be contracted with any trt
        # segment that depends on it.
        with g.control_dependencies([c]):
          d = self.trt_incompatible_op(n, name="incompatible")
        with g.control_dependencies([d]):
          n = math_ops.add(n, c, name="add")
          n = math_ops.mul(n, n, name="mul")
          n = math_ops.add(n, n, name="add1")
        n = self.trt_incompatible_op(n, name="incompatible1")
        with g.control_dependencies([d]):
          n = math_ops.add(n, c, name="add2")
          n = math_ops.mul(n, n, name="mul1")
          n = math_ops.add(n, n, name="add3")
      array_ops.squeeze(n, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[tuple(input_dims)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "my_trt_op_0": ["add", "add1", "mul"],
        "my_trt_op_1": ["add2", "add3", "mul1"]
    }


class ConstDataInputSingleEngineTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    input_name = "input"
    input_dims = [2, 32, 32, 3]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)
      with g.device("/GPU:0"):
        n = inp
        c = constant_op.constant(1.0, name="c")
        n = math_ops.add(n, c, name="add")
        n = math_ops.mul(n, n, name="mul")
        n = math_ops.add(n, n, name="add1")
      array_ops.squeeze(n, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[tuple(input_dims)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"my_trt_op_0": ["c", "add", "add1", "mul"]}


class ConstDataInputMultipleEnginesTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing multiple segment."""
    input_name = "input"
    input_dims = [2, 32, 32, 3]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)
      with g.device("/GPU:0"):
        n = inp
        c = constant_op.constant(1.0, name="c")
        n = math_ops.add(n, c, name="add")
        n = math_ops.mul(n, n, name="mul")
        n = math_ops.add(n, n, name="add1")
        n = self.trt_incompatible_op(n, name="incompatible1")
        n = math_ops.add(n, c, name="add2")
        n = math_ops.mul(n, n, name="mul1")
        n = math_ops.add(n, n, name="add3")
      array_ops.squeeze(n, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[tuple(input_dims)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "my_trt_op_0": ["add2", "add3", "mul1"],
        # Why segment ["add", "add1", "mul"] was assigned segment id 1
        # instead of 0: the parent node of this segment is actually const
        # node 'c', but it's removed later since it's const output of the
        # segment which is not allowed.
        "my_trt_op_1": ["add", "add1", "mul"]
    }


class ControlDependencyTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing multiple segment."""
    input_name = "input"
    input_dims = [2, 32, 32, 3]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)
      with g.device("/GPU:0"):
        c1 = constant_op.constant(1.0, name="c1")
        c2 = constant_op.constant(1.0, name="c2")
        d1 = constant_op.constant(1.0, name="d1")
        d2 = self.trt_incompatible_op(inp, name="d2")
        with g.control_dependencies([d1, d2]):
          add = math_ops.add(inp, c1, name="add")
        with g.control_dependencies([d1, d2]):
          mul = math_ops.mul(add, add, name="mul")
        with g.control_dependencies([d1, d2]):
          add1 = math_ops.add(mul, mul, name="add1")
        edge = self.trt_incompatible_op(add1, name="incompatible")
        with g.control_dependencies([d1, d2, add, mul]):
          add2 = math_ops.add(edge, c2, name="add2")
        with g.control_dependencies([d1, d2, add1, mul]):
          mul1 = math_ops.mul(add2, add2, name="mul1")
        with g.control_dependencies([d1, d2, add, add1]):
          add3 = math_ops.add(mul1, mul1, name="add3")
      array_ops.squeeze(add3, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[tuple(input_dims)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "my_trt_op_0": ["c1", "add", "add1", "mul"],
        "my_trt_op_1": ["c2", "add2", "add3", "mul1"]
    }


if __name__ == "__main__":
  test.main()
