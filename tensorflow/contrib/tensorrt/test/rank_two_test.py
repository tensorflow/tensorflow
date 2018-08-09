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
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RankTwoTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Test for rank 2 input in TF-TRT."""
    input_names = ["input", "input2"]
    input_dims = [[12, 5], [12, 5, 2, 2]]
    g = ops.Graph()
    with g.as_default():
      # Path 1 with rank 2 input
      outputs = []
      for i in range(2):
        x = array_ops.placeholder(
            dtype=dtypes.float32, shape=input_dims[i], name=input_names[i])
        c = constant_op.constant(1.0, name="c%d_1" % i)
        q = math_ops.add(x, c, name="add%d_1" % i)
        q = math_ops.abs(q, name="abs%d_1" % i)
        c = constant_op.constant(2.2, name="c%d_2" % i)
        q = math_ops.add(q, c, name="add%d_2" % i)
        q = math_ops.abs(q, name="abs%d_2" % i)
        c = constant_op.constant(3.0, name="c%d_3" % i)
        q = math_ops.add(q, c, name="add%d_3" % i)
        if i == 0:
          for j in range(2):
            q = array_ops.expand_dims(q, -1, name="expand%d_%d" % (i, j))
        q = gen_math_ops.reciprocal(q, name="reciprocal%d" % i)
        outputs.append(q)
      # Combine path 1 & 2
      q = math_ops.add(outputs[0], outputs[1], name="add")
      array_ops.squeeze(q, name=self.output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=input_names,
        input_dims=input_dims,
        expected_engines={
            "my_trt_op_0": [
                "add0_1", "add0_2", "add0_3", "c0_1", "c0_2", "c0_3", "abs0_1",
                "abs0_2"
            ],
            "my_trt_op_1": [
                "add", "add1_1", "add1_2", "add1_3", "c1_1", "c1_2", "c1_3",
                "abs1_1", "abs1_2", "reciprocal0", "reciprocal1"
            ],
        },
        expected_output_dims=tuple(input_dims[1]),
        allclose_atol=1.e-03,
        allclose_rtol=1.e-03)


if __name__ == "__main__":
  test.main()
