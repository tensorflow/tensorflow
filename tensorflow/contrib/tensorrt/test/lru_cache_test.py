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
"""Test LRUCache by running different input batch sizes on same network."""

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
from tensorflow.python.platform import test


class LRUCacheTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [[[1, 10, 10, 2]], [[2, 10, 10, 2]], [[4, 10, 10, 2]],
                  [[2, 10, 10, 2]]]
    expected_output_dims = [[[1, 10, 10, 1]], [[2, 10, 10, 1]], [[4, 10, 10,
                                                                  1]],
                            [[2, 10, 10, 1]]]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtype, shape=[None, 10, 10, 2], name=input_name)
      conv_filter = constant_op.constant(
          np.random.randn(3, 3, 2, 1), dtype=dtypes.float32)
      x = nn.conv2d(
          input=x,
          filter=conv_filter,
          strides=[1, 1, 1, 1],
          padding="SAME",
          name="conv")
      bias = constant_op.constant(
          np.random.randn(1, 10, 10, 1), dtype=dtypes.float32)
      x = math_ops.add(x, bias)
      x = nn.relu(x)
      x = array_ops.identity(x, name="output")
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=input_dims,
        output_names=[output_name],
        expected_output_dims=expected_output_dims)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

  def ShouldRunTest(self, run_params):
    return (run_params.dynamic_engine and
            not trt_test.IsQuantizationMode(run_params.precision_mode))


if __name__ == "__main__":
  test.main()
