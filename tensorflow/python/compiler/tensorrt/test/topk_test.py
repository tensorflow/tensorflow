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

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class TopKTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing Top-K in TF-TRT conversion."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 100]
    k = 5
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)
      k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
      values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
      values = array_ops.identity(values, name="output_values")
      indices = array_ops.identity(indices, name="output_indices")
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[[input_dims]],
        output_names=["output_values", "output_indices"],
        expected_output_dims=[[[100, k], [100, k]]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["Const", "TopK"]}


class TopKOutputTypeTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing that output type of engine using Top-K is set correctly."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 100]
    k = 5
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)
      k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
      values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
      # Reshape will act as a layer between the TopK output and the engine
      # output, requiring the output tensor of reshape to be set explicitly to
      # int32.
      indices = array_ops.reshape(indices, [100, 1, 5], name="Reshape")
      values = array_ops.identity(values, name="output_values")
      indices = array_ops.identity(indices, name="output_indices")
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[[input_dims]],
        output_names=["output_values", "output_indices"],
        expected_output_dims=[[[100, k], [100, 1, k]]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["Const", "TopK", "Reshape", "Reshape/shape"]}


if __name__ == "__main__":
  test.main()
