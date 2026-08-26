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
"""Tests for MirrorPad that run over tensors with more than 2**31 elements."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class LargeMirrorPadOpTest(test.TestCase):
  """Tests that belong in pad_op_test.py, but run over large tensors."""

  def testMirrorPadLargeTensor(self):
    # Regression test for GitHub issue 112304. The MirrorPad kernel used
    # 32-bit indexing unconditionally, so any tensor with more than
    # 2**31 - 1 elements was silently corrupted. Image i is constant-valued
    # (i % 251), so every padded image must also be constant.
    # CPU-only test, because it needs more memory than most GPUs have.
    num_images = 33000  # 33000 * 256 * 256 elements, just over 2**31.
    with ops.device("/cpu:0"):
      values = math_ops.cast(math_ops.range(num_images) % 251, dtypes.uint8)
      x = array_ops.broadcast_to(
          array_ops.reshape(values, [num_images, 1, 1, 1]),
          [num_images, 256, 256, 1])
      padded = array_ops.pad(
          x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
    self.assertEqual(padded.shape, [num_images, 258, 258, 1])
    # Probe images on both sides of the 2**31 element boundary, which falls
    # inside image 2**31 // (256 * 256) = 32768.
    for i in (0, 16000, 32767, 32768, 32999):
      image = self.evaluate(padded[i])
      self.assertAllEqual(
          image,
          np.full([258, 258, 1], i % 251, dtype=np.uint8),
          msg="image %d" % i)


if __name__ == "__main__":
  test.main()
