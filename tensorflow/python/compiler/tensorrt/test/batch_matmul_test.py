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

import unittest

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class BatchMatMultTestBase(trt_test.TfTrtIntegrationTestBase):
  """Base class for BatchMatMult tests."""

  # Shape inference of BatchMatMultV2 doesn't work. Use static batch size.
  def BuildParams(self, graph_fn, dtype, input_shapes, output_shapes):
    return self.BuildParamsWithMask(
        graph_fn=graph_fn,
        dtype=dtype,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        input_mask=[[True] * len(s) for s in input_shapes],
        output_mask=[[True] * len(s) for s in output_shapes],
        extra_inputs=[],
        extra_outputs=[])

  @classmethod
  def setUpClass(cls):
    if cls is BatchMatMultTestBase:
      raise unittest.SkipTest(
          "BatchMatMultTestBase defines base class for other test.")
    super(BatchMatMultTestBase, cls).setUpClass()


class BatchMatMulTwoTensorTest(BatchMatMultTestBase):
  """Testing conversion of BatchMatMul where both inputs are tensors."""

  def GraphFn(self, inp, inp1):
    x1 = math_ops.matmul(inp, inp1, name="matmul")
    # Relu to reach minimum segment size.
    x1 = nn.relu(x1, name="relu")
    return array_ops.identity(x1, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[12, 5, 8, 12], [12, 5, 12, 7]], [[12, 5, 8, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["matmul", "relu"]}


class BatchMatMulWeightBroadcastTest(BatchMatMultTestBase):
  """Testing BatchMatMulV2: one operand is weight and both have same rank."""

  def GraphFn(self, inp):
    dtype = inp.dtype
    b = constant_op.constant(
        np.random.randn(1, 5, 7), dtype=dtype, name="kernel")
    x1 = math_ops.matmul(inp, b, name="matmul")
    return array_ops.identity(x1, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 9, 5]],
                            [[12, 9, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["matmul", "kernel"]}


class BatchMatMulWeightBroadcastDims2Test(BatchMatMultTestBase):
  """Testing BatchMatMulV2: weight operand must be broadcasted."""

  def GraphFn(self, inp):
    dtype = inp.dtype
    b = constant_op.constant(np.random.randn(5, 7), dtype=dtype, name="kernel")
    x1 = math_ops.matmul(inp, b, name="matmul")
    return array_ops.identity(x1, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 9, 5]],
                            [[12, 9, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["matmul", "kernel"]}


if __name__ == "__main__":
  test.main()
