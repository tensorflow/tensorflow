# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for maximum spanning tree ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow_text.python.ops import mst_ops


class MstOpsTest(test.TestCase):
  """Testing rig."""

  @test_util.run_all_in_graph_and_eager_modes
  def testMaximumSpanningTree(self):
    """Tests that the MST op can recover a simple tree."""
    # The first batch element prefers 3 as root, then 3->0->1->2, for a total
    # score of 4+2+1=7.  The second batch element is smaller and has reversed
    # scores, so 0 is root and 0->2->1.
    num_nodes = constant_op.constant([4, 3], dtypes.int32)
    scores = constant_op.constant([[[0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [1, 2, 0, 0],
                                    [1, 2, 3, 4]],
                                   [[4, 3, 2, 9],
                                    [0, 0, 2, 9],
                                    [0, 0, 0, 9],
                                    [9, 9, 9, 9]]],
                                  dtypes.int32)  # pyformat: disable

    (max_scores, argmax_sources) = mst_ops.max_spanning_tree(
        num_nodes, scores, forest=False)

    self.assertAllEqual(max_scores, [7, 6])
    self.assertAllEqual(argmax_sources, [[3, 0, 1, 3],
                                         [0, 2, 0, -1]])  # pyformat: disable

  @test_util.run_deprecated_v1
  def testMaximumSpanningTreeGradient(self):
    """Tests the MST max score gradient."""
    with self.test_session() as session:
      num_nodes = constant_op.constant([4, 3], dtypes.int32)
      scores = constant_op.constant([[[0, 0, 0, 0],
                                      [1, 0, 0, 0],
                                      [1, 2, 0, 0],
                                      [1, 2, 3, 4]],
                                     [[4, 3, 2, 9],
                                      [0, 0, 2, 9],
                                      [0, 0, 0, 9],
                                      [9, 9, 9, 9]]],
                                    dtypes.int32)  # pyformat: disable

      mst_ops.max_spanning_tree(num_nodes, scores, forest=False, name='MST')
      mst_op = session.graph.get_operation_by_name('MST')

      d_loss_d_max_scores = constant_op.constant([3, 7], dtypes.float32)
      d_loss_d_num_nodes, d_loss_d_scores = (
          mst_ops.max_spanning_tree_gradient(mst_op, d_loss_d_max_scores))

      # The num_nodes input is non-differentiable.
      self.assertIs(d_loss_d_num_nodes, None)
      self.assertAllEqual(d_loss_d_scores.eval(),
                          [[[0, 0, 0, 3],
                            [3, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 0, 3]],
                           [[7, 0, 0, 0],
                            [0, 0, 7, 0],
                            [7, 0, 0, 0],
                            [0, 0, 0, 0]]])  # pyformat: disable

  @test_util.run_deprecated_v1
  def testMaximumSpanningTreeGradientError(self):
    """Numerically validates the max score gradient."""
    with self.test_session():
      # The maximum-spanning-tree-score function, as a max of linear functions,
      # is piecewise-linear (i.e., faceted).  The numerical gradient estimate
      # may be inaccurate if the epsilon ball used for the estimate crosses an
      # edge from one facet to another.  To avoid spurious errors, we manually
      # set the sample point so the epsilon ball fits in a facet.  Or in other
      # words, we set the scores so there is a non-trivial margin between the
      # best and second-best trees.
      scores_raw = [[[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [1, 2, 0, 0],
                     [1, 2, 3, 4]],
                    [[4, 3, 2, 9],
                     [0, 0, 2, 9],
                     [0, 0, 0, 9],
                     [9, 9, 9, 9]]]  # pyformat: disable

      # Use 64-bit floats to reduce numerical error.
      scores = constant_op.constant(scores_raw, dtypes.float64)
      init_scores = np.array(scores_raw)

      num_nodes = constant_op.constant([4, 3], dtypes.int32)
      max_scores = mst_ops.max_spanning_tree(num_nodes, scores, forest=False)[0]

      gradient_error = test.compute_gradient_error(scores, [2, 4, 4],
                                                   max_scores, [2], init_scores)
      self.assertIsNot(gradient_error, None)


if __name__ == '__main__':
  test.main()
