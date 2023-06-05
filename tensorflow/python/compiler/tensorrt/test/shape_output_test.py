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

import os

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ShapeOutputTest(trt_test.TfTrtIntegrationTestBase):
  """Test shape value output with TF-TRT."""

  def setUp(self):
    super().setUp()
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
      return ["TRTEngineOp_000", "TRTEngineOp_001"]
    else:
      # Second segment not converted in implicit batch mode, because its
      # tensors have only one dimensions
      return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # We cannot calibrate without bulding the engine, we turn of INT8 test.
    return (run_params.dynamic_shape and
            run_params.precision_mode != "INT8", "no calibration dynamic shape")


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
    return ["TRTEngineOp_000", "TRTEngineOp_001"]


class PrunedInputTest(trt_test.TfTrtIntegrationTestBase):
  """In TRT 7, an input tensor can be pruned if it is not used by the network.

  This happens if only its shape is used, but the shape is already defined by
  the optimization profile by setting min=max. (nvbugs/3153064)

  After pruning, the TRT network has no input bindings.
  """

  def setUp(self):
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    q = array_ops.shape(x)
    q = q * 2 + q * q
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 2, 5, 3]], [[4]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True]],
        output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # Shape op is only converted in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")


class PrunedInputTest2(trt_test.TfTrtIntegrationTestBase):
  """Two inputs, one of the is pruned."""

  def setUp(self):
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x, y):
    q = array_ops.shape(x)
    z = y * y + y
    z = gen_array_ops.reshape(z, q)
    out_0 = array_ops.identity(q, name="output_0")
    out_1 = array_ops.identity(z, name="output_1")
    return (out_0, out_1)

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 2, 5, 3], [2, 15]], [[4], [1, 2, 5, 3]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True], [False, True]],
        output_mask=[[True], [False, True, True, True]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # Shape op is only converted in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")


class ShapeValueMaskTest(trt_test.TfTrtIntegrationTestBase):
  """Confirm that 0D and 1D non int tensors are not treated as shape tensors."""

  def setUp(self):
    super().setUp()
    # This is to test whether shape value mask is correctly set in case engine
    # construction has failed.
    os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "True"
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"

  def tearDown(self):
    super().tearDown()
    os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "False"
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"

  def GraphFn(self, x, y):
    q = 2 * x + y
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float16, [[3], []], [[3]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[True], []],
        output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    if run_params.dynamic_shape:
      return ["TRTEngineOp_000"]
    else:
      return []

  def ShouldRunTest(self, run_params):
    # We cannot calibrate without bulding the engine, we turn of INT8 test.
    return (run_params.dynamic_shape and
            run_params.precision_mode != "INT8", "no calibration dynamic shape")


class InputProfile(trt_test.TfTrtIntegrationTestBase):
  """The shape profiles has to fit values of shape tensors, but for regular

  tensors the values do not matter. Here we test shape profile managment with
  an INT32 input tensor that is not a shape tensor. The extra inputs with
  dim=10 would trigger an error if we mistakenly treat it as shape tensors.
  """

  def setUp(self):
    super().setUp()

    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    z = x * x + x + 1
    z = array_ops.identity(z, name="output_0")
    return z

  def GetParams(self):
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.int32,
        [[4]],
        [[4]],
        extra_inputs=[[[5]], [[10]]],
        extra_outputs=[[[5]], [[10]]],
        input_mask=[[False]],
        output_mask=[[False]],
    )

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # Shape op is only converted in dynamic shape mode.
    return (
        run_params.dynamic_shape
        and run_params.is_v2
        and not trt_test.IsQuantizationMode(run_params.precision_mode),
        "Test v2 dynamic_shapes without INT8",
    )


if __name__ == "__main__":
  test.main()
