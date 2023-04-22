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

import os
import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


class BinaryTensorWeightBroadcastTest(trt_test.TfTrtIntegrationTestBase):
  """Tests for scale & elementwise layers in TF-TRT."""

  def _ConstOp(self, shape):
    return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)

  def GraphFn(self, x):
    for weights_shape in [
        (1,),  # scale
        (24, 1, 1),  # scale
        (24, 24, 20),  # scale
        (20,),  # elementwise
        (1, 24, 1, 1),  # elementwise
        (1, 24, 24, 1),  # elementwise
        (1, 24, 24, 20),  # elementwise
        (24, 20),  # elementwise
    ]:
      a = self._ConstOp(weights_shape)
      f = x + a
      x = self.trt_incompatible_op(f)
      a = self._ConstOp(weights_shape)
      f = a + x
      x = self.trt_incompatible_op(f)
    return gen_array_ops.reshape(x, [5, -1], name="output_0")

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[10, 24, 24, 20]],
                            [[5, 23040]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_%d" % i for i in range(16)]

  # TODO(b/176540862): remove this routine to disallow native segment execution
  # for TensorRT 7+.
  def setUp(self):
    super(trt_test.TfTrtIntegrationTestBase, self).setUp()  # pylint: disable=bad-super-call
    if trt_test.IsTensorRTVersionGreaterEqual(7):
      os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"

if __name__ == "__main__":
  test.main()
