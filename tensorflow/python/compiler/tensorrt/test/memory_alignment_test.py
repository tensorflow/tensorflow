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
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class MemoryAlignmentTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of BatchMatMul in TF-TRT conversion."""

  def GraphFn(self, inp):
    dtype = inp.dtype
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
    return array_ops.squeeze(out, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 15, 15, 3]],
                            [[2, 15, 15, 10]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-06 if run_params.precision_mode == "FP32" else 1.e-02

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 0.1


if __name__ == "__main__":
  test.main()
