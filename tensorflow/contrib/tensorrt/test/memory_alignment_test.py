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

import numpy as np

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class MemoryAlignmentTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing conversion of BatchMatMul in TF-TRT conversion."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [2, 15, 15, 3]
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        e1 = constant_op.constant(
            np.random.randn(1, 1, 3, 5), name="kernel_1", dtype=dtype)
        e2 = constant_op.constant(
            np.random.randn(1, 1, 5, 10), name="kernel_2", dtype=dtype)
        conv = nn.conv2d(
            input=inp,
            filter=e1,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        out = nn.conv2d(
            input=conv,
            filter=e2,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_2")
      array_ops.squeeze(out, name=self.output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        expected_engines=["my_trt_op_0"],
        expected_output_dims=(2, 15, 15, 10),
        allclose_atol=1.e-02,
        allclose_rtol=1.e-02)


if __name__ == "__main__":
  test.main()
