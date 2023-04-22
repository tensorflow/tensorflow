# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test conversion of graphs involving INT32 tensors and operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CastInt32ToFp32Test(trt_test.TfTrtIntegrationTestBase):
  """Tests cast to FP32 are splitted in FP16 mode."""

  def _ConstOp(self, shape, dtype):
    return constant_op.constant(np.random.randn(*shape), dtype=dtype)

  def GraphFn(self, x):
    b_f = self._ConstOp((1, 10), dtypes.float32)
    x_f = math_ops.cast(x, dtypes.float32)
    x_f = math_ops.mul(x_f, b_f)
    b_f = self._ConstOp((1, 10), dtypes.float32)
    x_f = math_ops.add(x_f, b_f)
    return array_ops.identity(x_f, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.int32, [[1, 10]], [[1, 10]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    if run_params.precision_mode == "FP16":
      return {"TRTEngineOp_0": ["Cast", "Add", "Mul"]}
    else:
      return {"TRTEngineOp_0": ["Add", "Mul"]}

if __name__ == "__main__":
  test.main()
