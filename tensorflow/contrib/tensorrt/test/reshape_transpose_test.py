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


class SimpleReshapeTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [-1, 24*24*2])
        # Add identities to ensure we have at least min_segment_size=3 nodes
        identity = array_ops.identity(reshape, "identity")
        identity = array_ops.identity(identity, "identity2")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 24*24*2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

class ReshapeToScalarTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [1]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=input_dims, name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [])
        # Add identities to ensure we have at least min_segment_size=3 nodes
        identity = array_ops.identity(reshape, "identity")
        identity = array_ops.identity(identity, "identity2")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[()])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # No engine should be created so exclude INT8 to avoid "ERROR:tensorflow:Not
    # a calib graph. Doesn't seem to contain any calibration nodes.""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and 
            not run_params.dynamic_engine)

class ReshapeBatchDimensionTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [2, 50, 24, 24, 2])
        # Add identities to ensure we have at least min_segment_size=3 nodes
        identity = array_ops.identity(reshape, "identity")
        identity = array_ops.identity(identity, "identity2")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(2, 50, 24, 24, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # No engine should be created so exclude INT8 to avoid "ERROR:tensorflow:Not
    # a calib graph. Doesn't seem to contain any calibration nodes.""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and 
            not run_params.dynamic_engine)

class ReshapeBatchDimensionTest2(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [-1, 50, 24, 24, 2])
        # Add identities to ensure we have at least min_segment_size=3 nodes
        identity = array_ops.identity(reshape, "identity")
        identity = array_ops.identity(identity, "identity2")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(2, 50, 24, 24, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # No engine should be created so exclude INT8 to avoid "ERROR:tensorflow:Not
    # a calib graph. Doesn't seem to contain any calibration nodes.""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and 
            not run_params.dynamic_engine)

class ReshapeBatchDimensionTest3(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [2, 50, -1, 24, 2])
        # Add identities to ensure we have at least min_segment_size=3 nodes
        identity = array_ops.identity(reshape, "identity")
        identity = array_ops.identity(identity, "identity2")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(2, 50, 24, 24, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # No engine should be created so exclude INT8 to avoid "ERROR:tensorflow:Not
    # a calib graph. Doesn't seem to contain any calibration nodes.""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and 
            not run_params.dynamic_engine)

class ReshapeInverseTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [-1, 24*24*2])
        reshape = array_ops.reshape(reshape, [-1, 24, 24, 2])
        identity = array_ops.identity(reshape, "identity")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 24, 24, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

class ManyReshapeTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        reshape = array_ops.reshape(inp, [-1, 24*24, 2])
        reshape = array_ops.reshape(reshape, [-1, 24*2, 24])
        reshape = array_ops.reshape(reshape, [-1, 24, 24*2])
        reshape = array_ops.reshape(reshape, [-1, 6, 4, 24, 2])
        reshape = array_ops.reshape(reshape, [-1, 6, 4, 6, 4, 2])
        reshape = array_ops.reshape(reshape, [-1, 6, 4, 6, 4, 2, 1])
        reshape = array_ops.reshape(reshape, [-1, 24, 24, 2])
        identity = array_ops.identity(reshape, "identity")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 24, 24, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

class SimpleTransposeTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        # to NCHW
        transpose = array_ops.transpose(inp, [0, 3, 1, 2])
        identity = array_ops.identity(transpose, "identity")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 2, 24, 24)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

class TransposeBatchDimensionTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        # to NCHW
        transpose = array_ops.transpose(inp, [2, 1, 0, 3])
        identity = array_ops.identity(transpose, "identity")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(24, 24, 100, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # No engine should be created so exclude INT8 to avoid "ERROR:tensorflow:Not
    # a calib graph. Doesn't seem to contain any calibration nodes.""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and 
            not run_params.dynamic_engine)

class TransposeInverseTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        # to NCHW
        transpose = array_ops.transpose(inp, [0, 3, 1, 2])
        # back to NHWC
        transpose = array_ops.transpose(transpose, [0, 2, 3, 1])
        identity = array_ops.identity(transpose, "identity")
      array_ops.identity(identity, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(100, 24, 24, 2)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_0"]

if __name__ == "__main__":
  test.main()
