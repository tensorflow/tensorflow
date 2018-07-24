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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RankTwoTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Test for rank 2 input in TF-TRT."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [12, 5]
    input2_name = "input2"
    input2_dims = [12, 5, 2, 2]
    g = ops.Graph()
    with g.as_default():
      # path 1 with rank 2 input
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)
      q = x + 1.0
      q = math_ops.abs(q)
      q = q + 2.2
      q = math_ops.abs(q)
      q = q + 3.0
      q = array_ops.expand_dims(q, -1)
      q = array_ops.expand_dims(q, -1)
      a = gen_math_ops.reciprocal(q)
      # path 2 with rank 4 input
      x = array_ops.placeholder(dtype=dtype, shape=input2_dims, name=input2_name)
      q = x + 1.0
      q = math_ops.abs(q)
      q = q + 2.2
      q = math_ops.abs(q)
      q = q + 3.0
      b = gen_math_ops.reciprocal(q)
      # combine path 1 & 2
      q = a + b
      array_ops.squeeze(q, name=self.output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name, input2_name],
        input_dims=[input_dims, input2_dims],
        num_expected_engines=2,
        expected_output_dims=(12, 5, 2, 2),
        allclose_atol=1.e-03,
        allclose_rtol=1.e-03)


if __name__ == "__main__":
  test.main()
