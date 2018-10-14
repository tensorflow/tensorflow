# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Clip Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import numerics
from tensorflow.python.platform import test


class ClipOpsTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(ClipOpsTest, self).__init__(method_name)

  def _testClipByNorm(self, inputs, max_norm, expected):
    with self.cached_session() as sess:
      input_op = constant_op.constant(inputs)
      clipped = clip_ops.clip_by_norm(input_op, max_norm)
      check_op = numerics.add_check_numerics_ops()
      result, _ = sess.run([clipped, check_op])
    self.assertAllClose(result, expected)

  def testClipByNorm(self):
    # Simple example
    self._testClipByNorm([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 4.0,
                         [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]])
    # Zero norm
    self._testClipByNorm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 4.0,
                         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


if __name__ == "__main__":
  test.main()
