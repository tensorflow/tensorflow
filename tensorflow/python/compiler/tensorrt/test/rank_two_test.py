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

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RankTwoTest(trt_test.TfTrtIntegrationTestBase):
  """Test for rank 2 input in TF-TRT."""

  def GraphFn(self, x1, x2):
    # Two paths: first with rank 2 input, second with rank 4 input.
    outputs = []
    xs = [x1, x2]
    for i in range(2):
      x = xs[i]
      c = constant_op.constant(1.0, name="c%d_1" % i)
      q = math_ops.add(x, c, name="add%d_1" % i)
      q = math_ops.abs(q, name="abs%d_1" % i)
      c = constant_op.constant(2.2, name="c%d_2" % i)
      q = math_ops.add(q, c, name="add%d_2" % i)
      q = math_ops.abs(q, name="abs%d_2" % i)
      c = constant_op.constant(3.0, name="c%d_3" % i)
      q = math_ops.add(q, c, name="add%d_3" % i)
      if i == 0:
        axis = constant_op.constant(-1, dtype=dtypes.int32, name="axis")
        for j in range(2):
          q = array_ops.expand_dims(q, axis, name="expand%d_%d" % (i, j))
        q = self.trt_incompatible_op(q)
      q = gen_math_ops.reciprocal(q, name="reciprocal%d" % i)
      outputs.append(q)
    # Combine both paths
    q = math_ops.add(outputs[0], outputs[1], name="add")
    return array_ops.squeeze(q, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[12, 5], [12, 5, 2, 2]], [[12, 5, 2, 2]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    expected_engines = {
        "TRTEngineOp_000": [
            "add0_1", "add0_2", "add0_3", "c0_1", "c0_2", "c0_3", "abs0_1",
            "abs0_2", "expand0_0", "expand0_1", "axis"
        ],
        "TRTEngineOp_001": [
            "add1_1", "add1_2", "add1_3", "c1_1", "c1_2", "c1_3", "abs1_1",
            "abs1_2", "reciprocal1"
        ]
    }
    if not run_params.dynamic_shape:
      # The two ops can't be in the same cluster as the ops in TRTEngineOp_0
      # due to trt_incompatible_op. They can't be in the same cluster as the
      # ops in TRTEngineOP_1 because their batch size belongs to a different
      # equivalent class.
      expected_engines["TRTEngineOp_002"] = ["add", "reciprocal0"]
    else:
      # In dynamic shape mode the batch size of the ops can differ,
      # therefore the final ops will be merged to TRTEngineOP_1.
      expected_engines["TRTEngineOp_001"] += ["add", "reciprocal0"]

    return expected_engines


if __name__ == "__main__":
  test.main()
