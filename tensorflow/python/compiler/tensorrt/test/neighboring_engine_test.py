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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class NeighboringEngineTest(trt_test.TfTrtIntegrationTestBase):
  """Neighboring node wiring tests in TF-TRT conversion."""

  def GraphFn(self, x):
    dtype = x.dtype
    e = constant_op.constant(
        np.random.normal(.3, 0.05, [3, 2, 3, 4]), name="weights", dtype=dtype)
    conv = nn.conv2d(
        input=x,
        filter=e,
        data_format="NCHW",
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    b = constant_op.constant(
        np.random.normal(1.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    t = math_ops.mul(conv, b, name="mul")
    e = self.trt_incompatible_op(conv, name="incompatible")
    t = math_ops.sub(t, e, name="sub")
    return array_ops.squeeze(t, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 3, 7, 5]],
                            [[2, 4, 5, 4]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_0": ["bias", "mul", "sub"],
        "TRTEngineOp_1": ["weights", "conv"]
    }


if __name__ == "__main__":
  test.main()
