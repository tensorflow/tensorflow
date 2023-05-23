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
"""Model script to test TF-TensorRT integration with data dependent shapes"""

import os

from unittest import SkipTest  # pylint: disable=g-importing-member

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TrtModeTestBase(trt_test.TfTrtIntegrationTestBase):
  """Creates a network that has data dependent shapes."""

  def GraphFn(self, x):
    # With the unique() op we create a tensor with data dependent shape.
    x = math_ops.floor(x * 10)
    y, idx = array_ops.unique(x)
    y = y * 2 + y

    # The rest is only needed to ensure that the output has the same size as
    # the input (expected by the test harness).
    padding = array_ops.constant([0])
    n = array_ops.shape(x) - array_ops.shape(y)
    padding = array_ops.concat([padding, n], 0)
    padding = array_ops.expand_dims(padding, 0)
    y = array_ops.pad(y, padding)

    return array_ops.identity(y, name="output_0")

  def ShouldRunTest(self, run_params):
    # We have a single dimension, and that is changed, therefore the graph can
    # only be converted in dynamic shape mode.
    return (run_params.dynamic_engine and run_params.is_v2 and
            run_params.dynamic_shape and
            run_params.use_calibration, "test v2 dynamic engine and "
            "calibration")

  def setUp(self):
    super().setUp()
    # The input shape depends on random input data. It can happen that we do
    # not have engine for the actual shape. Therefore we enable native segment
    # execution.
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"

  def tearDown(self):
    super().tearDown()
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[12]], [[12]])

  def ExpectedEnginesToBuild(self, run_params):
    return ["TRTEngineOp_000", "TRTEngineOp_001"]


if __name__ == "__main__":
  test.main()
