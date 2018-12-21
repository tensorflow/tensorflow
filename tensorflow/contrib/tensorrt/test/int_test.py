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
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class BiasaddMatMulTest(trt_test.TfTrtIntegrationTestBase):

  def _ConstOp(self, shape, dtype):
    return constant_op.constant(np.random.randn(*shape), dtype=dtype)

  def GetParams(self):
    """Testing conversion of BiasAdd MatMul in TF-TRT conversion."""
    # Note that tf.nn.bias_add supports up to 5 dimensions.
    input_dims = [100, 4]
    g = ops.Graph()
    with g.as_default():
      # float part
      # input1 = array_ops.placeholder(
      #     dtype=dtypes.float32, shape=input_dims, name='input_float32')
      # b1 = self._ConstOp((4, 10), dtypes.float32)
      # x1 = math_ops.matmul(input1, b1)
      # b1 = self._ConstOp((1, 10), dtypes.float32)
      # x1 = x1 + b1

      # int part
      input2 = array_ops.placeholder(
          dtype=dtypes.int32, shape=input_dims, name='input_int32')
      b2 = self._ConstOp((4, 10), dtypes.int32)
      x2 = math_ops.matmul(input2, b2)
      b2 = self._ConstOp((10,), dtypes.int32)
      x2 = nn.bias_add(x2, b2)

      # combine
      #y = x1 * x2

      #out1 = array_ops.identity(x1, name='output_float32')
      out2 = array_ops.identity(x2, name='output_int32')
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=['input_int32'],
        input_dims=[[100, 4]],
        output_names=['output_int32'],
        expected_output_dims=[(100, 10)])

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    conversion_params = super(BiasaddMatMulTest,
                              self).GetConversionParams(run_params)
    return conversion_params._replace(
        max_batch_size=100,
        maximum_cached_engines=1,
        # Disable layout optimizer, since it will convert BiasAdd with NHWC
        # format to NCHW format under four dimentional input.
        rewriter_config=trt_test.OptimizerDisabledRewriterConfig())

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return []

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # TODO(aaroey): Trt 4.0 forbids conversion for tensors with rank <3 in int8
    # mode, which is a bug. Re-enable this when trt library is fixed.
    return not trt_test.IsQuantizationMode(run_params.precision_mode) #and not run_params.dynamic_engine


if __name__ == "__main__":
  test.main()
