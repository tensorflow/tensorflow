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

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class BiasaddMatMulTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of BiasAdd MatMul in TF-TRT conversion."""

  def _ConstOp(self, shape):
    return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)

  def GraphFn(self, x):
    input_matrix_rows = 4
    input_matrix_columns = 144

    b = self._ConstOp((input_matrix_columns, 4))
    x1 = math_ops.matmul(x, b)
    b = self._ConstOp((1, 4))
    x1 = x1 + b

    b = self._ConstOp((input_matrix_rows, 144))
    x2 = self.trt_incompatible_op(x)
    x2 = math_ops.matmul(x2, b, transpose_a=True)
    x2 = gen_array_ops.reshape(x2, [4, -1])
    x2 = self.trt_incompatible_op(x2)

    b = self._ConstOp((4, input_matrix_columns))
    x3 = math_ops.matmul(x, b, transpose_b=True)

    b = self._ConstOp((16, input_matrix_rows))
    x4 = self.trt_incompatible_op(x)
    x4 = math_ops.matmul(x4, b, transpose_b=True, transpose_a=True)
    x4 = gen_array_ops.reshape(x4, [4, -1])
    x4 = self.trt_incompatible_op(x4)

    # Note that tf.nn.bias_add supports up to 5 dimensions.
    b = self._ConstOp((input_matrix_columns, 48))
    x5 = math_ops.matmul(x, b)
    b = self._ConstOp((48,))
    x5 = nn.bias_add(x5, b)
    # TODO(b/154672994): Put the reshape back when the bug is fixed.
    # x5 = gen_array_ops.reshape(x5, [4, -1])

    x6 = gen_array_ops.reshape(x, [4, 24, 6])
    b = self._ConstOp((6,))
    x6 = nn.bias_add(x6, b, data_format="NHWC")
    x6 = gen_array_ops.reshape(x6, [4, -1])

    x7 = gen_array_ops.reshape(x, [4, 12, 4, 3])
    b = self._ConstOp((3,))
    x7 = nn.bias_add(x7, b, data_format="NHWC")
    x7 = gen_array_ops.reshape(x7, [4, -1])

    x8 = gen_array_ops.reshape(x, [4, 4, 3, 2, 6])
    b = self._ConstOp((6,))
    x8 = nn.bias_add(x8, b, data_format="NHWC")
    x8 = gen_array_ops.reshape(x8, [4, -1])

    x9 = gen_array_ops.reshape(x, [4, 12, 3, 2, 2])
    b = self._ConstOp((12,))
    x9 = nn.bias_add(x9, b, data_format="NCHW")
    x9 = gen_array_ops.reshape(x9, [4, -1])

    x10 = gen_array_ops.reshape(x, [4, 3, 4, 12])
    b = self._ConstOp((3,))
    x10 = nn.bias_add(x10, b, data_format="NCHW")
    x10 = gen_array_ops.reshape(x10, [4, -1])

    x11 = gen_array_ops.reshape(x, [4, 6, 24])
    b = self._ConstOp((6,))
    x11 = nn.bias_add(x11, b, data_format="NCHW")
    x11 = gen_array_ops.reshape(x11, [4, -1])

    out = array_ops.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11],
                           axis=-1)
    return array_ops.squeeze(out, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[4, 144]],
                            [[4, 6680]])

  def setUp(self):
    super(trt_test.TfTrtIntegrationTestBase, self).setUp()  # pylint: disable=bad-super-call
    # Disable layout optimizer, since it will convert BiasAdd with NHWC
    # format to NCHW format under four dimentional input.
    self.DisableNonTrtOptimizers()

  def GetMaxBatchSize(self, run_params):
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None

    return 4

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    if run_params.dynamic_shape:
      # Increased conversion rate in dynamic shape mode due to a few additional
      # conversions for MatMul, Reshape and Concat ops. This increases the size
      # of the candidate segments and results in two more TrtEngineOps.
      return ["TRTEngineOp_0", "TRTEngineOp_1", "TRTEngineOp_2"]
    else:
      return ["TRTEngineOp_0"]


if __name__ == "__main__":
  test.main()
