# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.kernels.bcast_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops.gen_array_ops import _broadcast_gradient_args


class BcastOpsTest(tf.test.TestCase):

  def _GetGradientArgs(self, xs, ys):
    with self.test_session() as sess:
      return sess.run(_broadcast_gradient_args(xs, ys))

  def testBasic(self):
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

    r0, r1 = self._GetGradientArgs([1], [2, 3, 5])
    self.assertAllEqual(r0, [0, 1, 2])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1])

    r0, r1 = self._GetGradientArgs([5], [2, 3, 5])
    self.assertAllEqual(r0, [0, 1])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [3, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0])

    r0, r1 = self._GetGradientArgs([3, 5], [2, 3, 5])
    self.assertAllEqual(r0, [0])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 3, 5], [3, 1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 2])

    r0, r1 = self._GetGradientArgs([3, 1], [2, 3, 5])
    self.assertAllEqual(r0, [0, 2])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 1, 5], [3, 1])
    self.assertAllEqual(r0, [1])
    self.assertAllEqual(r1, [0, 2])

    r0, r1 = self._GetGradientArgs([3, 1], [2, 1, 5])
    self.assertAllEqual(r0, [0, 2])
    self.assertAllEqual(r1, [1])

  def testZeroDims(self):
    r0, r1 = self._GetGradientArgs([2, 0, 3, 0, 5], [3, 0, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1])

    r0, r1 = self._GetGradientArgs([3, 0, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r0, [0, 1])
    self.assertAllEqual(r1, [])

    r0, r1 = self._GetGradientArgs([2, 0, 3, 0, 5], [3, 1, 5])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 3])

    r0, r1 = self._GetGradientArgs([3, 1, 5], [2, 0, 3, 0, 5])
    self.assertAllEqual(r0, [0, 1, 3])
    self.assertAllEqual(r1, [])


if __name__ == "__main__":
  tf.test.main()
