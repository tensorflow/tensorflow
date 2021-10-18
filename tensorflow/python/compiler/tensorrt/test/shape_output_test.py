# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ShapeOutputTest(trt_test.TfTrtIntegrationTestBase):
  """Test shape value output with TF-TRT."""

  def setUp(self):
    super(trt_test.TfTrtIntegrationTestBase, self).setUp()  # pylint: disable=bad-super-call
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    # The first engine returns the shape of q, which equals the shape of x. The
    # values of x are actually not needed for the TRT engine. This way x is
    # neither a shape tensor nor an execution tensor, we still need to set its
    # shape (binding dimensions). We confirm with this test that the binding
    # dimensions of x are correctly set before we execute the engine.
    q = 2 * x + 1
    q = array_ops.shape(q)
    q = math_ops.cast(q, dtypes.float32)
    q = self.trt_incompatible_op(q)
    q = q * 2 + q * q
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[2, 2, 5, 3]], [[4]],
        extra_inputs=[[[8, 2, 5, 3]]],
        extra_outputs=[[[4]]],
        input_mask=[[False, True, True, True]],
        output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    if run_params.dynamic_shape:
      return ["TRTEngineOp_0", "TRTEngineOp_1"]
    else:
      # Second segment not converted in implicit batch mode, because its
      # tensors have only one dimensions
      return ["TRTEngineOp_0"]


class ShapeOutputWithSingleInputProfile(ShapeOutputTest):
  """Same as the previous test, but with a single input profile."""

  def setUp(self):
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[2, 2, 5, 3]], [[4]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True]],
        output_mask=[[True]])


class ShapeOutputWithSingleInputAndReshape(trt_test.TfTrtIntegrationTestBase):
  """Similar to the previous test, but the ShapeOp output is reshaped to 2D.

  This makes the output tensor not compatible with shape tensor.
  """

  def setUp(self):
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    q = 2 * x + 1
    q = array_ops.shape(q)
    q = gen_array_ops.reshape(q, [2, 2])
    q = math_ops.cast(q, dtypes.float32)
    q = self.trt_incompatible_op(q)
    q = q * 2 + q * q
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[2, 2, 5, 3]], [[2, 2]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True]],
        output_mask=[[True, True]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    return ["TRTEngineOp_0", "TRTEngineOp_1"]


if __name__ == "__main__":
  test.main()
