# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class BestSplitsClassificationTests(test_util.TensorFlowTestCase):

  def setUp(self):
    self.finished = [3, 5]
    self.node_map = [-1, -1, -1, 0, -1, 3, -1, -1, -1]
    self.candidate_counts = [[[153., 50., 60., 40., 3.],
                              [200., 70., 30., 70., 30.]],
                             [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
                             [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
                             [[40., 10., 10., 10., 10.],
                              [30., 10., 5., 5., 10.]]]
    self.total_counts = [[400., 100., 100., 100., 100.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [400., 100., 100., 100., 100.]]
    self.squares = []

  def testSimple(self):
    with self.test_session():
      split_indices = tensor_forest_ops.best_splits(
          self.finished,
          self.node_map,
          self.candidate_counts,
          self.squares,
          self.total_counts,
          self.squares,
          regression=False)

      self.assertAllEqual([0, 1], split_indices.eval())

  def testNoFinished(self):
    with self.test_session():
      split_indices = tensor_forest_ops.best_splits(
          [],
          self.node_map,
          self.candidate_counts,
          self.squares,
          self.total_counts,
          self.squares,
          regression=False)

      self.assertAllEqual([], split_indices.eval())

  def testBadInput(self):
    del self.total_counts[1]

    with self.test_session():
      with self.assertRaisesOpError(
          'Number of accumulators should be the same in split_sums '
          'and accumulator_sums.'):
        tensor_forest_ops.best_splits(
            self.finished,
            self.node_map,
            self.candidate_counts,
            self.squares,
            self.total_counts,
            self.squares,
            regression=False).eval()


class BestSplitsRegressionTests(test_util.TensorFlowTestCase):

  def setUp(self):
    self.finished = [3, 5]
    self.node_map = [-1, -1, -1, 0, -1, 3, -1, -1, -1]
    self.candidate_sums = [[[5., 8., 8., 8.], [5., 10., 10., 10.]],
                           [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                           [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                           [[10., 10., 20., 10.], [10., 5., 5., 5.]]]

    self.candidate_squares = [[[5., 50., 50., 50.], [5., 50., 50., 50.]],
                              [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                              [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                              [[10., 40., 50., 60.], [10., 40., 40., 40.]]]

    self.total_sums = [[15., 10., 10., 10.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [20., 20., 20., 20.]]

    self.total_squares = [[15., 50., 50., 50.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [20., 60., 60., 60.]]

  def testSimple(self):
    with self.test_session():
      split_indices = tensor_forest_ops.best_splits(
          self.finished,
          self.node_map,
          self.candidate_sums,
          self.candidate_squares,
          self.total_sums,
          self.total_squares,
          regression=True)

      self.assertAllEqual([1, 0], split_indices.eval())


if __name__ == '__main__':
  googletest.main()
