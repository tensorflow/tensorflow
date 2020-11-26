# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for where op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
# pylint: enable=g-direct-tensorflow-import


class WhereOpTest(xla_test.XLATestCase):

  def testWhere(self):
    """Test first form of where (return indices)."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        true_vals = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [[True, False, False], [False, True, True]]
      self.assertAllEqual([[0, 0], [1, 1], [1, 2]],
                          sess.run(true_vals, {x: feed}))

  def testWhereGather(self):
    """Test where followed by a gather."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        value = array_ops.constant([[0, 1], [2, 3]], dtypes.float32)
        true_vals = array_ops.where(x)

        # Gather 0, 2, 3.
        gathered = array_ops.gather_nd(value, true_vals)

      feed = [[True, False], [True, True]]
      self.assertAllEqual([0, 2, 3], sess.run(gathered, {x: feed}))

  def testWhereGatherReduce(self):
    """Test where followed by a gather and a reduce."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        value = array_ops.constant([[0, 1], [2, 3]], dtypes.float32)
        indices = array_ops.where(x)

        # Reduce to 5
        gathered = array_ops.gather_nd(value, indices)
        reduction = math_ops.reduce_sum(gathered)

      feed = [[True, False], [True, True]]
      self.assertAllEqual(5, sess.run(reduction, {x: feed}))

  def testWhere1D(self):
    """Test first form of where (return indices)."""

    with self.session() as sess:
      with self.test_scope():
        x = array_ops.placeholder(dtypes.bool)
        result = array_ops.where(x)

      # Output of the computation is dynamic.
      feed = [True, False, True]
      self.assertAllEqual([[0], [2]], sess.run(result, {x: feed}))


if __name__ == "__main__":
  test.main()
