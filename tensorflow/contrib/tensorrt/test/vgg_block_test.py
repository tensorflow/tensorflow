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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test
from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test


class VGGBlockTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """single vgg layer test in TF-TRT conversion"""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [5, 8, 8, 2]
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtype, shape=input_dims, name=input_name)
      x, mean_x, var_x = nn_impl.fused_batch_norm(
          x,
          np.random.randn(2).astype(np.float32),
          np.random.randn(2).astype(np.float32),
          mean=np.random.randn(2).astype(np.float32),
          variance=np.random.randn(2).astype(np.float32),
          is_training=False)
      e = constant_op.constant(
          np.random.randn(1, 1, 2, 6), name="weights", dtype=dtype)
      conv = nn.conv2d(
          input=x, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
      b = constant_op.constant(
          np.random.randn(6), name="bias", dtype=dtype)
      t = nn.bias_add(conv, b, name="biasAdd")
      relu = nn.relu(t, "relu")
      idty = array_ops.identity(relu, "ID")
      v = nn_ops.max_pool(
          idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
      array_ops.squeeze(v, name="output")
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        num_expected_engines=1,
        expected_output_dims=(5, 2, 2, 6),
        allclose_atol=1.e-03,
        allclose_rtol=1.e-03)

if __name__ == "__main__":
  test.main()
