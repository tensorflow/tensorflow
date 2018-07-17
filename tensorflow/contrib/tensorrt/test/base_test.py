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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test


class SimpleSingleEngineGraphDefTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    # TODO(aaroey): test graph with different dtypes.
    dtype = dtypes.float32
    input_dims = [100, 24, 24, 2]
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=self.input_name)
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
      array_ops.squeeze(pool, name=self.output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_dims=input_dims,
        num_expected_engines=1,
        expected_output_dims=(100, 6, 6, 6),
        allclose_atol=1.e-03,
        allclose_rtol=1.e-03)


class SimpleMultiEngineGraphDefTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing multiple segment."""
    # TODO(aaroey): test graph with different dtypes.
    dtype = dtypes.float32
    input_dims = [100, 24, 24, 2]
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=self.input_name)
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
            np.random.randn(input_dims[0], 12, 12, 6), dtype=dtype)
        p = conv * c1
        c2 = constant_op.constant(
            np.random.randn(input_dims[0], 12, 12, 6), dtype=dtype)
        q = conv / c2

        edge = self.trt_incompatible_op(q)
        edge /= edge
        r = edge + edge

        p -= edge
        q *= edge
        s = p + q
        s -= r
      array_ops.squeeze(s, name=self.output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_dims=input_dims,
        num_expected_engines=2,
        expected_output_dims=(100, 12, 12, 6),
        allclose_atol=1.e-03,
        allclose_rtol=1.e-03)


# TODO(aaroey): add a large complex graph to test.

if __name__ == "__main__":
  test.main()
