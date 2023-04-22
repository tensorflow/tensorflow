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
"""Model script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import SkipTest  # pylint: disable=g-importing-member

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TrtModeTestBase(trt_test.TfTrtIntegrationTestBase):
  """Test squeeze on batch dim and some unary operations in TF-TRT."""

  def GraphFn(self, x1):
    q = math_ops.abs(x1)
    q = q + 1.0
    q = q * 3.0
    q = array_ops.squeeze(q, 0)
    q = math_ops.abs(q)
    q = q + 5.0
    return array_ops.identity(q, name="output_0")

  def ShouldRunTest(self, run_params):
    # Squeeze op produces dynamic shaped values. Therefore, we don't run the
    # test with static engine to avoid native segment execution.
    return (run_params.dynamic_engine, "test dynamic engine only")

  def GetParams(self):
    """The input has 1 as a first dimension, which is removed by the squeeze.

    op in the graph.

    In explicit batch mode, TensorRT can convert the whole graph. In this mode
    it is possible to manipulate the batch dimension using the squeeze op.

    In implicit batch mode TensorRT cannot convert the whole graph. We are not
    allowed to manipulate (squeeze) the first dimension in implicit batch mode.
    Therefore the graph will be converted using multiple segments.
    """
    return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 12, 5]],
                            [[12, 5]])

  @classmethod
  def setUpClass(cls):
    if cls is TrtModeTestBase:
      raise SkipTest("TrtModeTestBase defines base class for other test.")
    super(TrtModeTestBase, cls).setUpClass()


class ImplicitBatchTest(TrtModeTestBase):

  def GetMaxBatchSize(self, run_params):
    if run_params.dynamic_engine:
      return None

    # The first dimension of the input is squeezed and the batch size for the
    # rest OPs is 12.
    return 12

  def ExpectedEnginesToBuild(self, run_params):
    """Check that the expected engine is built.

    Args:
      run_params: the run parameters.

    Returns:
      the expected engines to build.

    The squeeze op is not converted by TensorRT in implicit batch mode.
    Because of this we have two TRTEngineOp in the graphs: one for the
    subgraph before 'squeeze(q,0)', and another one for the rest of the ops
    after the 'squeeze(q,0)'.
    """
    return ["TRTEngineOp_0", "TRTEngineOp_1"]


class ExplicitBatchTest(TrtModeTestBase):

  def GetParams(self):
    """We specify input/output masks with static (known) shapes."""
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 12, 5]], [[12, 5]],
        input_mask=[[True, True, True]],
        output_mask=[[True, True]],
        extra_inputs=[],
        extra_outputs=[])

  def ExpectedEnginesToBuild(self, run_params):
    """Check that the expected engine is built.

    Args:
      run_params: the run parameters.

    Returns:
      the expected engines to build.

    In explicit batch mode the whole graph is converted using a single engine.
    """
    return ["TRTEngineOp_0"]

  def ShouldRunTest(self, run_params):
    # Only run for TRT 6 and above.
    return run_params.is_v2 and trt_test.IsTensorRTVersionGreaterEqual(6) and (
        not run_params.use_calibration), "test v2, >=TRT6 and non-calibration"

  def setUp(self):
    super().setUp()
    self.SetDynamicShapeModeAndProfileStrategy(
        profile_strategy="ImplicitBatchModeCompatible")


class DynamicShapesTest(TrtModeTestBase):
  """Test with dynamic input shapes.

  DynamicShapesTest is different from ExplicitBatchTest in that it uses input
  and output masks to change the input and output shapes to unknown shapes.
  """

  def GetParams(self):
    """We specify input/output mask with dynamic (unknown) shapes.

    A single
    engine with three optimization profiles can handle the three different
    input shapes.
    """
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 12, 5]], [[12, 5]],
        extra_inputs=[[[1, 2, 3]], [[1, 4, 6]]],
        extra_outputs=[[[2, 3]], [[4, 6]]],
        input_mask=[[False, False, False]],
        output_mask=[[False, False]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

  def ShouldRunTest(self, run_params):
    # Only run for TRT 6 and above.
    return run_params.is_v2 and trt_test.IsTensorRTVersionGreaterEqual(6) and (
        not run_params.use_calibration), "test v2 >=TRT6 and non-calibration"

  def setUp(self):
    super().setUp()
    self.SetDynamicShapeModeAndProfileStrategy(
        profile_strategy="ImplicitBatchModeCompatible")


if __name__ == "__main__":
  test.main()
