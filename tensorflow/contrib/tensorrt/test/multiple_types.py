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
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      input1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name='input_float32')

      b1 = self._ConstOp((4, 10), dtypes.float32)
      x1 = math_ops.matmul(input1, b)
      b1 = self._ConstOp((1, 10), dtypes.float32)
      x1 = x1 + b1

      input2 = array_ops.placeholder(
          dtype=dtypes.int32, shape=input_dims, name='input_int32')

      b2 = self._ConstOp((4, 10), dtypes.float32)
      x2 = math_ops.matmul(input2, b)
      b2 = self._ConstOp((1, 10), dtypes.float32)
      x2 = x2 + b2
      out = array_ops.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11],
                             axis=-1)
      out = array_ops.squeeze(out, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(4, 6680)])

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    conversion_params = super(BiasaddMatMulTest,
                              self).GetConversionParams(run_params)
    return conversion_params._replace(
        max_batch_size=4,
        maximum_cached_engines=1,
        # Disable layout optimizer, since it will convert BiasAdd with NHWC
        # format to NCHW format under four dimentional input.
        rewriter_config=trt_test.OptimizerDisabledRewriterConfig())

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # TODO(aaroey): Trt 4.0 forbids conversion for tensors with rank <3 in int8
    # mode, which is a bug. Re-enable this when trt library is fixed.
    return not trt_test.IsQuantizationMode(run_params.precision_mode)


if __name__ == "__main__":
  test.main()
