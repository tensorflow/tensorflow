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

  def GetParams(self):
    """Testing conversion of BiasAdd MatMul in TF-TRT conversion."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [48, 12]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)

      b = constant_op.constant(np.random.randn(12, 4), dtype=dtype)
      x1 = math_ops.matmul(x, b)
      b = constant_op.constant(np.random.randn(1, 4), dtype=dtype)
      x1 = x1 + b

      b = constant_op.constant(np.random.randn(48, 4), dtype=dtype)
      x2 = math_ops.matmul(x, b, transpose_a=True)
      x2 = gen_array_ops.reshape(x2, [48, 1])

      b = constant_op.constant(np.random.randn(4, 12), dtype=dtype)
      x3 = math_ops.matmul(x, b, transpose_b=True)

      b = constant_op.constant(np.random.randn(16, 48), dtype=dtype)
      x4 = math_ops.matmul(x, b, transpose_b=True, transpose_a=True)
      x4 = gen_array_ops.reshape(x4, [48, 4])

      x5 = gen_array_ops.reshape(x, [4, 144])
      b = constant_op.constant(np.random.randn(144, 48), dtype=dtype)
      x5 = math_ops.matmul(x5, b)
      b = constant_op.constant(np.random.randn(48), dtype=dtype)
      x5 = nn.bias_add(x5, b)
      x5 = gen_array_ops.reshape(x5, [48, 4])

      x6 = gen_array_ops.reshape(x, [4, 12, 12])
      b = constant_op.constant(np.random.randn(12), dtype=dtype)
      x6 = nn.bias_add(x6, b, data_format="NHWC")
      x6 = gen_array_ops.reshape(x6, [48, -1])

      x7 = gen_array_ops.reshape(x, [4, 12, 3, 4])
      b = constant_op.constant(np.random.randn(4), dtype=dtype)
      x7 = nn.bias_add(x7, b, data_format="NHWC")
      x7 = gen_array_ops.reshape(x7, [48, -1])

      x8 = gen_array_ops.reshape(x, [4, 12, 3, 2, 2])
      b = constant_op.constant(np.random.randn(2), dtype=dtype)
      x8 = nn.bias_add(x8, b, data_format="NHWC")
      x8 = gen_array_ops.reshape(x8, [48, -1])

      x9 = gen_array_ops.reshape(x, [4, 12, 3, 2, 2])
      b = constant_op.constant(np.random.randn(3), dtype=dtype)
      x9 = nn.bias_add(x9, b, data_format="NCHW")
      x9 = gen_array_ops.reshape(x9, [48, -1])

      x10 = gen_array_ops.reshape(x, [4, 12, 3, 4])
      b = constant_op.constant(np.random.randn(12), dtype=dtype)
      x10 = nn.bias_add(x10, b, data_format="NCHW")
      x10 = gen_array_ops.reshape(x10, [48, -1])

      x11 = gen_array_ops.reshape(x, [4, 12, 12])
      b = constant_op.constant(np.random.randn(4), dtype=dtype)
      x11 = nn.bias_add(x11, b, data_format="NCHW")
      x11 = gen_array_ops.reshape(x11, [48, -1])

      out = array_ops.concat(
          [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
      out = array_ops.squeeze(out, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(48, 89)])

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    return super(BiasaddMatMulTest,
                 self).GetConversionParams(run_params)._replace(
                     max_batch_size=48, maximum_cached_engines=2)

  def _ValidEngines(self):
    """Engines expected to build and run."""
    return [
        "my_trt_op_0", "my_trt_op_1", "my_trt_op_2", "my_trt_op_6",
        "my_trt_op_7", "my_trt_op_8", "my_trt_op_9"
    ]

  def _InvalidEngines(self):
    """Engines that will cause conversion error at building time."""
    return ["my_trt_op_3", "my_trt_op_4", "my_trt_op_5"]

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    # In dynamic engine mode the engines are built in execution time, not in
    # conversion time, so build errors occurs later. Here three of the engines
    # will be failed to built but the corresponding engine op are still created.
    # TODO(aaroey, jjsjann123): fix this.
    if (run_params.dynamic_engine and
        not trt_test.IsQuantizationMode(run_params.precision_mode)):
      return self._ValidEngines() + self._InvalidEngines()
    return self._ValidEngines()

  def ExpectedEnginesToRun(self, run_params):
    """Return the expected engines to run."""
    return self._ValidEngines()

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # TODO(aaroey): Trt 4.0 forbids conversion for tensors with rank <3 in int8
    # mode, which is a bug. Re-enable this when trt library is fixed.
    return not trt_test.IsQuantizationMode(run_params.precision_mode)


if __name__ == "__main__":
  test.main()
