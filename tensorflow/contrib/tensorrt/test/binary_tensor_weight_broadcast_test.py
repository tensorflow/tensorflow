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

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BinaryTensorWeightBroadcastTest(trt_test.TfTrtIntegrationTestBase):

  def _ConstOp(self, shape):
    return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)

  def GetParams(self):
    """Tests for scale & elementwise layers in TF-TRT."""
    input_name = "input"
    input_dims = [10, 24, 24, 20]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)
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
      gen_array_ops.reshape(x, [5, -1], name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(5, 23040)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["my_trt_op_%d" % i for i in range(16)]


if __name__ == "__main__":
  test.main()
