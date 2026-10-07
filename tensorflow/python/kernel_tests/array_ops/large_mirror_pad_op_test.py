# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for MirrorPad Op."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class LargeMirrorPadOpTest(test.TestCase):
  """Tests that belong in pad_op_test.py, but run over large tensors."""

  def testMirrorPadLargeTensor(self):
    # Regression test for a silent wrong result when the padded output has
    # more than 2**31 elements. MirrorPad used to hand its tensors to Eigen
    # through To32Bit() unconditionally, so the output element count was
    # truncated to int32 and the assignment loop ran the wrong number of
    # iterations, leaving part or all of the output unwritten.
    #
    # Rank 5 keeps the input small: REFLECT requires each padding to be less
    # than its dimension, so 24 on both sides of a dimension of 25 gives 73,
    # and 25 on both sides of a dimension of 26 gives 76. That is
    # 73**4 * 76 = 2,158,266,316 output elements, over 2**31 by 10,782,668,
    # from a 10 MB input.
    #
    # CPU-only test, because the output alone needs 2.2 GB.
    with ops.device("/cpu:0"):
      x = array_ops.ones([25, 25, 25, 25, 26], dtype=dtypes.uint8)
      padded = array_ops.pad(x, [[24, 24]] * 4 + [[25, 25]], mode="REFLECT")
    with self.session(use_gpu=False):
      self.assertEqual([73] * 4 + [76], padded.shape.as_list())
      # Reflecting only ever copies existing elements, so every element of
      # the result must be the 1 it was built from.
      self.assertEqual(1, self.evaluate(math_ops.reduce_min(padded)))
      self.assertEqual(1, self.evaluate(math_ops.reduce_max(padded)))


if __name__ == "__main__":
  test.main()
