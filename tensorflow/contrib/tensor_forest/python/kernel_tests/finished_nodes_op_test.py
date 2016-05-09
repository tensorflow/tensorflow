# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for tf.contrib.tensor_forest.ops.finished_nodes_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow  # pylint: disable=unused-import

from tensorflow.contrib.tensor_forest.python.ops import training_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class FinishedNodesTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.leaves = [1, 3, 4]
    self.node_map = [-1, -1, -1, 0, 1, -1]
    self.pcw_total_splits = [[6, 3, 3], [11, 4, 7], [0, 0, 0], [0, 0, 0],
                             [0, 0, 0]]
    self.ops = training_ops.Load()

  def testSimple(self):
    with self.test_session():
      finished = self.ops.finished_nodes(self.leaves, self.node_map,
                                         self.pcw_total_splits,
                                         num_split_after_samples=10)

      self.assertAllEqual([4], finished.eval())

  def testNoAccumulators(self):
    with self.test_session():
      finished = self.ops.finished_nodes(self.leaves, [-1] * 6,
                                         self.pcw_total_splits,
                                         num_split_after_samples=10)

      self.assertAllEqual([], finished.eval())

  def testBadInput(self):
    with self.test_session():
      with self.assertRaisesOpError(
          'leaf_tensor should be one-dimensional'):
        finished = self.ops.finished_nodes([self.leaves], self.node_map,
                                           self.pcw_total_splits,
                                           num_split_after_samples=10)

        self.assertAllEqual([], finished.eval())

if __name__ == '__main__':
  googletest.main()
