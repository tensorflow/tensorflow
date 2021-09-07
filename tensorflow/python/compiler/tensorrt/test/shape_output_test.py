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

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ShapeOutputTest(trt_test.TfTrtIntegrationTestBase):
  """Test shape value output with TF-TRT."""

  def GraphFn(self, x):
   q = 2 * x + 1
   q = array_ops.shape(q)
   q = math_ops.cast(q, dtypes.float32)
   q = self.trt_incompatible_op(q)
   q = q * 2 + q * q
   return array_ops.identity(q, name="output_0")

  def GetParams(self):
    return self.BuildParamsWithMask(self.GraphFn, dtypes.float32,
                            [[1, 2, 5, 3]], [[4]],
                            extra_inputs=[], # [[[1, 1, 2, 3]]],
                            extra_outputs=[], #[[[4]]],
                            input_mask=[[False, False, False, False]],
                            output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0", "TRTEngineOp_1"]

  def setUp(self):
    super().setUp()
    self.SetDynamicShapeModeAndProfileStrategy(
        profile_strategy="ImplicitBatchModeCompatible")


if __name__ == "__main__":
  test.main()
