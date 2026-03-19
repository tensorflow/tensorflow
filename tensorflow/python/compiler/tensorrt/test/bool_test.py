# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BoolTest(trt_test.TfTrtIntegrationTestBase):
  """Test for boolean operations in TF-TRT."""

  def GraphFn(self, x1, x2):
    x = math_ops.logical_and(x1, x2)
    x = math_ops.logical_or(x, x2)
    q = math_ops.not_equal(x, x2)
    q = math_ops.logical_not(q)
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    shape = [2, 32, 32, 3]
    return self.BuildParams(self.GraphFn, dtypes.bool, [shape, shape], [shape])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    reason = "Boolean ops are not implemented "
    return (run_params.dynamic_shape, reason + "in ImplicitBatch mode") \
        if trt_utils.is_linked_tensorrt_version_greater_equal(8, 2, 0)  \
        else (False, reason + "for TRT < 8.2.0")


if __name__ == "__main__":
  test.main()
