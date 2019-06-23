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

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BatchMatMulTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of BatchMatMul in TF-TRT conversion."""

  def GraphFn(self, inp, inp1, inp2):
    dtype = inp.dtype
    b = constant_op.constant(np.random.randn(12, 5, 12, 7), dtype=dtype)
    x1 = math_ops.matmul(inp, b)
    c = constant_op.constant(np.random.randn(5, 1, 1), dtype=dtype)
    x1 = x1 + c

    x2 = math_ops.matmul(inp, inp1)
    d = constant_op.constant(np.random.randn(5, 1, 1), dtype=dtype)
    x2 = x2 * d

    e = self.trt_incompatible_op(inp)
    e = gen_array_ops.reshape(e, [12, 40, 12])
    x3 = math_ops.matmul(e, inp2)
    f = constant_op.constant(np.random.randn(40, 1), dtype=dtype)
    x3 = x3 + f
    x3 = gen_array_ops.reshape(x3, [12, 5, 8, 7])
    x3 = self.trt_incompatible_op(x3)

    out = x1 + x2 + x3
    return array_ops.squeeze(out, name="output_0")

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[12, 5, 8, 12], [12, 5, 12, 7], [12, 12, 7]],
                            [[12, 5, 8, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0", "TRTEngineOp_1", "TRTEngineOp_2"]


if __name__ == "__main__":
  test.main()
