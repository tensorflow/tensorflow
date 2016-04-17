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
"""Tests for tf.contrib.tensor_forest.ops.best_splits_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow  # pylint: disable=unused-import

from tensorflow.contrib.tensor_forest.python.ops import training_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class BestSplitsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.finished = [3, 5]
    self.node_map = [-1, -1, -1, 0, -1, 3, -1, -1, -1]
    self.candidate_counts = [[[50., 60., 40., 3.], [70., 30., 70., 30.]],
                             [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                             [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                             [[10., 10., 10., 10.], [10., 5., 5., 10.]]]
    self.total_counts = [[100., 100., 100., 100.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [100., 100., 100., 100.]]
    self.ops = training_ops.Load()

  def testSimple(self):
    with self.test_session():
      split_indices = self.ops.best_splits(
          self.finished, self.node_map, self.candidate_counts,
          self.total_counts)

      self.assertAllEqual([0, 1], split_indices.eval())

  def testNoFinished(self):
    with self.test_session():
      split_indices = self.ops.best_splits(
          [], self.node_map, self.candidate_counts, self.total_counts)

      self.assertAllEqual([], split_indices.eval())

  def testBadInput(self):
    del self.total_counts[1]

    with self.test_session():
      with self.assertRaisesOpError(
          'Number of accumulators should be the same in pcw_candidate_splits '
          'and pcw_total_splits.'):
        self.ops.best_splits(
            self.finished, self.node_map, self.candidate_counts,
            self.total_counts).eval()


if __name__ == '__main__':
  googletest.main()
