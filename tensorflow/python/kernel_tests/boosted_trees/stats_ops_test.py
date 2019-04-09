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
"""Tests for boosted_trees stats kernels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.platform import googletest


class StatsOpsTest(test_util.TensorFlowTestCase):
  """Tests stats_ops."""

  def testCalculateBestGainsWithoutRegularization(self):
    """Testing Gain calculation without any regularization."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = [
          [
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
              [[0., 0.], [-.33, .58], [0., 0.], [.3, .4]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 0
          [
              [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.3, .5], [-.05, .06], [.06, .07]],  # node 1
              [[.1, .1], [.2, .3], [-.4, .5], [.07, .08]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 1
      ]  # num_features * shape=[max_splits, num_buckets, 2]

      (node_ids_list, gains_list, thresholds_list, left_node_contribs_list,
       right_node_contribs_list
      ) = boosted_trees_ops.calculate_best_gains_per_feature(
          node_id_range,
          stats_summary_list,
          l1=0.0,
          l2=0.0,
          tree_complexity=0.0,
          min_node_weight=0,
          max_splits=max_splits)

      self.assertAllEqual([[1, 2], [1, 2]], self.evaluate(node_ids_list))
      self.assertAllClose([[0.004775, 0.41184], [0.02823, 0.41184]],
                          self.evaluate(gains_list))
      self.assertAllEqual([[1, 1], [1, 1]], self.evaluate(thresholds_list))
      # The left node contrib will be later added to the previous node value to
      # make the left node value, and the same for right node contrib.
      self.assertAllClose([[[-.416667], [.568966]], [[-.6], [-.75]]],
                          self.evaluate(left_node_contribs_list))
      self.assertAllClose([[[-.592593], [-.75]], [[-.076923], [.568966]]],
                          self.evaluate(right_node_contribs_list))

  def testCalculateBestGainsWithL2(self):
    """Testing Gain calculation with L2."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = [
          [
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
              [[0., 0.], [-.33, .58], [0., 0.], [.3, .4]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 0
          [
              [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.3, .5], [-.05, .06], [.06, .07]],  # node 1
              [[.1, .1], [.2, .3], [-.4, .5], [.07, .08]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 1
      ]  # num_features * shape=[max_splits, num_buckets, 2]

      (node_ids_list, gains_list, thresholds_list, left_node_contribs_list,
       right_node_contribs_list
      ) = boosted_trees_ops.calculate_best_gains_per_feature(
          node_id_range,
          stats_summary_list,
          l1=0.0,
          l2=0.1,
          tree_complexity=0.0,
          min_node_weight=0,
          max_splits=max_splits)

      self.assertAllEqual([[1, 2], [1, 2]], self.evaluate(node_ids_list))
      self.assertAllClose([[0., 0.33931375], [0.01879096, 0.33931375]],
                          self.evaluate(gains_list))
      self.assertAllEqual([[0, 1], [1, 1]], self.evaluate(thresholds_list))
      # The left node contrib will be later added to the previous node value to
      # make the left node value, and the same for right node contrib.
      self.assertAllClose([[[0.], [.485294]], [[-.5], [-.6]]],
                          self.evaluate(left_node_contribs_list))
      self.assertAllClose([[[-.424658], [-.6]], [[-.043478], [.485294]]],
                          self.evaluate(right_node_contribs_list))

  def testCalculateBestGainsWithL1(self):
    """Testing Gain calculation with L1."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = [
          [
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
              [[0., 0.], [-.33, .58], [0., 0.], [.3, .4]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 0
          [
              [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.3, .5], [-.05, .06], [.06, .07]],  # node 1
              [[.1, .1], [.2, .3], [-.4, .5], [.07, .08]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 1
      ]  # num_features * shape=[max_splits, num_buckets, 2]

      l1 = 0.1
      (node_ids_list, gains_list, thresholds_list, left_node_contribs_list,
       right_node_contribs_list
      ) = boosted_trees_ops.calculate_best_gains_per_feature(
          node_id_range,
          stats_summary_list,
          l1=l1,
          l2=0.0,
          tree_complexity=0.0,
          min_node_weight=0,
          max_splits=max_splits)

      self.assertAllEqual([[0, 1], [1, 1]], self.evaluate(thresholds_list))

      self.assertAllEqual([[1, 2], [1, 2]], self.evaluate(node_ids_list))
      self.assertAllClose([[[0.0], [0.3965517]], [[-0.4], [-0.5]]],
                          self.evaluate(left_node_contribs_list))

      self.assertAllClose([[[-0.3333333], [-0.5]], [[0.0], [0.396552]]],
                          self.evaluate(right_node_contribs_list))

      # Gain should also include an adjustment of the gradient by l1.
      self.assertAllClose([[0.0, 0.191207], [0.01, 0.191207]],
                          self.evaluate(gains_list))

  def testCalculateBestGainsWithTreeComplexity(self):
    """Testing Gain calculation with L2."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = [
          [
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
              [[0., 0.], [-.33, .58], [0., 0.], [.3, .4]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 0
          [
              [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.3, .5], [-.05, .06], [.06, .07]],  # node 1
              [[.1, .1], [.2, .3], [-.4, .5], [.07, .08]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 1
      ]  # num_features * shape=[max_splits, num_buckets, 2]

      l2 = 0.1
      tree_complexity = 3.
      (node_ids_list, gains_list, thresholds_list, left_node_contribs_list,
       right_node_contribs_list
      ) = boosted_trees_ops.calculate_best_gains_per_feature(
          node_id_range,
          stats_summary_list,
          l1=0.0,
          l2=l2,
          tree_complexity=tree_complexity,
          min_node_weight=0,
          max_splits=max_splits)

      self.assertAllEqual([[1, 2], [1, 2]], self.evaluate(node_ids_list))

      self.assertAllClose([[-3., -2.66068625], [-2.98120904, -2.66068625]],
                          self.evaluate(gains_list))

      self.assertAllEqual([[0, 1], [1, 1]], self.evaluate(thresholds_list))
      # The left node contrib will be later added to the previous node value to
      # make the left node value, and the same for right node contrib.
      self.assertAllClose([[[0.], [.485294]], [[-.5], [-.6]]],
                          self.evaluate(left_node_contribs_list))
      self.assertAllClose([[[-.424658], [-.6]], [[-.043478], [.485294]]],
                          self.evaluate(right_node_contribs_list))

  def testCalculateBestGainsWithMinNodeWeight(self):
    """Testing Gain calculation without any regularization."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = [
          [
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.15, .036], [.06, .07], [.1, .2]],  # node 1
              [[0., 0.], [-.33, .68], [0., 0.], [.3, .4]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 0
          [
              [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.3, .5], [-.05, .6], [.06, .07]],  # node 1
              [[.1, .1], [.2, .03], [-.4, .05], [.07, .08]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 1
      ]  # num_features * shape=[max_splits, num_buckets, 2]

      (node_ids_list, gains_list, thresholds_list, left_node_contribs_list,
       right_node_contribs_list
      ) = boosted_trees_ops.calculate_best_gains_per_feature(
          node_id_range,
          stats_summary_list,
          l1=0.0,
          l2=0.0,
          tree_complexity=0.0,
          min_node_weight=1,
          max_splits=max_splits)

      # We can't split node 1 on feature 1 and node 2 on feature 2 because of
      # the min node weight.
      self.assertAllEqual([[2], [1]], self.evaluate(node_ids_list))
      self.assertAllClose([[0.384314], [0.098013]], self.evaluate(gains_list))
      self.assertAllEqual([[1], [1]], self.evaluate(thresholds_list))
      self.assertAllClose([[[0.4852941]], [[-.6]]],
                          self.evaluate(left_node_contribs_list))
      self.assertAllClose([[[-0.75]], [[-0.014925]]],
                          self.evaluate(right_node_contribs_list))

  def testCalculateBestGainsWithMinNodeWeightNoSplitOnFeturePossible(self):
    """Testing Gain calculation without any regularization."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = [
          [
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.15, .0036], [.06, .007], [.1, .2]],  # node 1
              [[0., 0.], [-.33, .068], [0., 0.], [.3, .04]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 0
          [
              [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
              [[0., 0.], [.3, .5], [-.05, .6], [.06, .07]],  # node 1
              [[.1, .1], [.2, .03], [-.4, .05], [.07, .08]],  # node 2
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
          ],  # feature 1
      ]  # num_features * shape=[max_splits, num_buckets, 2]

      (node_ids_list, _, _, _,
       _) = boosted_trees_ops.calculate_best_gains_per_feature(
           node_id_range,
           stats_summary_list,
           l1=0.0,
           l2=0.0,
           tree_complexity=0.0,
           min_node_weight=1,
           max_splits=max_splits)

      # We can't split either of the nodes on the first feature
      self.assertEqual(2, len(self.evaluate(node_ids_list)))
      self.assertAllEqual([], self.evaluate(node_ids_list)[0])
      self.assertAllEqual([1], self.evaluate(node_ids_list)[1])

      # Now check when we can't split on any feature
      (node_ids_list, _, _, _,
       _) = boosted_trees_ops.calculate_best_gains_per_feature(
           node_id_range,
           stats_summary_list,
           l1=0.0,
           l2=0.0,
           tree_complexity=0.0,
           min_node_weight=10,
           max_splits=max_splits)
      self.assertAllEqual([[], []], self.evaluate(node_ids_list))

  @test_util.run_deprecated_v1
  def testMakeStatsSummarySimple(self):
    """Simple test for MakeStatsSummary."""
    expected_stats_summary = np.asarray([1., 5., 2., 6., 3., 7., 4., 8.])
    self.assertAllClose(
        expected_stats_summary.reshape((1, 2, 2, 2)),
        boosted_trees_ops.make_stats_summary(
            node_ids=[0, 0, 1, 1],
            gradients=[[1.], [2.], [3.], [4.]],
            hessians=[[5.], [6.], [7.], [8.]],
            bucketized_features_list=[[0, 1, 0, 1]],
            max_splits=2,
            num_buckets=2))

  @test_util.run_deprecated_v1
  def testAggregateStatsSimple(self):
    # Get the same result as MakeStatsSummary Op.
    expected_stats_summary = np.asarray([1., 5., 2., 6., 3., 7., 4., 8.])
    # shape=[max_splits, num_buckets, feature_dim, stats_dim]
    expected_stats_summary = np.reshape(expected_stats_summary, (2, 2, 1, 2))
    # Reshape feature dim and bucket id axes
    expected_stats_summary = np.swapaxes(expected_stats_summary, 1, 2)
    self.assertAllClose(
        expected_stats_summary,
        boosted_trees_ops.boosted_trees_aggregate_stats(
            node_ids=[0, 0, 1, 1],
            gradients=[[1.], [2.], [3.], [4.]],
            hessians=[[5.], [6.], [7.], [8.]],
            feature=[[0], [1], [0], [1]],
            max_splits=2,
            num_buckets=2))

  def testMakeStatsSummaryAccumulate(self):
    """Tests that Summary actually accumulates."""
    with self.cached_session():
      max_splits = 3
      num_buckets = 4
      node_ids = [1, 1, 2, 2, 1, 1, 2, 0]
      gradients = [[.1], [.2], [.3], [-.4], [-.05], [.06], [.07], [.08]]
      hessians = [[.2], [.3], [.4], [.5], [.06], [.07], [.08], [.09]]

      # Tests a single feature.
      bucketized_features = [[3, 1, 2, 0, 1, 2, 0, 1]]
      result = boosted_trees_ops.make_stats_summary(
          node_ids, gradients, hessians, bucketized_features, max_splits,
          num_buckets)  # shape=[max_splits, num_buckets, num_features, 2]
      self.assertAllClose(
          [[
              [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0
              [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
              [[-.33, .58], [0., 0.], [.3, .4], [0., 0.]],  # node 2
          ]],
          self.evaluate(result))

  def testAggregateStatsAccumulate(self):
    """Tests that Summary actually accumulates."""
    max_splits = 3
    num_buckets = 4
    node_ids = [1, 1, 2, 2, 1, 1, 2, 0]
    gradients = [[.1], [.2], [.3], [-.4], [-.05], [.06], [.07], [.08]]
    hessians = [[.2], [.3], [.4], [.5], [.06], [.07], [.08], [.09]]

    # Tests a single feature.
    bucketized_features = [[3], [1], [2], [0], [1], [2], [0], [1]]
    result = boosted_trees_ops.boosted_trees_aggregate_stats(
        node_ids, gradients, hessians, bucketized_features, max_splits,
        num_buckets)
    # shape=[max_splits, num_buckets, feature_dim, stats_dim]
    # Get the same result as MakeStatsSummary Op.
    expected_stats_summary = [
        [[[0., 0.]], [[.08, .09]], [[0., 0.]], [[0., 0.]]],
        [[[0., 0.]], [[.15, .36]], [[.06, .07]], [[.1, .2]]],
        [[[-.33, .58]], [[0., 0.]], [[.3, .4]], [[0., 0.]]],
    ]
    # Swap feature dim and bucket id axis
    expected_stats_summary = np.swapaxes(expected_stats_summary, 1, 2)
    self.assertAllClose(expected_stats_summary, result)

  def testMakeStatsSummaryMultipleFeatures(self):
    """Tests that MakeStatsSummary works for multiple features."""
    with self.cached_session():
      max_splits = 3
      num_buckets = 4
      node_ids = [1, 1, 2, 2, 1, 1, 2, 0]
      gradients = [[.1], [.2], [.3], [-.4], [-.05], [.06], [.07], [.08]]
      hessians = [[.2], [.3], [.4], [.5], [.06], [.07], [.08], [.09]]

      # Tests multiple features.
      # The output from another feature will stored be in 3rd dimension.
      bucketized_features = [[3, 1, 2, 0, 1, 2, 0, 1], [0, 0, 0, 2, 2, 3, 3, 2]]
      result = boosted_trees_ops.make_stats_summary(
          node_ids, gradients, hessians, bucketized_features, max_splits,
          num_buckets)  # shape=[max_splits, num_buckets, num_features, 2]
      self.assertAllClose(
          [
              [
                  [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0
                  [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
                  [[-.33, .58], [0., 0.], [.3, .4], [0., 0.]],  # node 2
              ],  # feature 0
              [
                  [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0
                  [[.3, .5], [0., 0.], [-.05, .06], [.06, .07]],  # node 1
                  [[.3, .4], [0., 0.], [-.4, .5], [.07, .08]],  # node 2
              ],  # feature 1
          ],
          self.evaluate(result))

  def testAggregatesSummaryMultipleDimensionFeature(self):
    """Tests that MakeStatsSummary works for multiple features."""
    expected_stats_summary = np.asarray(
        [[0, 0, 0, 0, .08, .09, 0, 0, 0, 0, .08, .09, 0, 0, 0, 0],
         [0, 0, .3, .5, .15, .36, 0, 0, .06, .07, -.05, .06, .1, .2, .06, .07],
         [-.33, .58, .3, .4, 0, 0, 0, 0, .3, .4, -.4, .5, 0, 0, .07, .08]])
    with self.cached_session():
      max_splits = 3
      num_buckets = 4
      node_ids = [1, 1, 2, 2, 1, 1, 2, 0]
      gradients = [[.1], [.2], [.3], [-.4], [-.05], [.06], [.07], [.08]]
      hessians = [[.2], [.3], [.4], [.5], [.06], [.07], [.08], [.09]]

      # Tests multiple features.
      bucketized_features = [[3, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 3],
                             [0, 3], [1, 2]]
      result = boosted_trees_ops.boosted_trees_aggregate_stats(
          node_ids, gradients, hessians, bucketized_features, max_splits,
          num_buckets)
      # Reshape to [max_splits, num_buckets, feature_dim, stats_dim]
      expected_stats_summary = np.reshape(expected_stats_summary, (3, 4, 2, 2))
      # Swap feature_dim and bucket_id axis
      expected_stats_summary = np.swapaxes(expected_stats_summary, 1, 2)
      self.assertAllClose(expected_stats_summary, result)

  def testAggregateStatsMultiClass(self):
    """Tests that Summary actually accumulates."""
    with self.cached_session():
      max_splits = 3
      num_buckets = 4
      node_ids = [1, 1, 2, 2, 1, 1, 2, 0]
      gradients = [[.1, .2], [.2, .4], [.3, .6], [-.4, -.8], [-.05, -.1],
                   [.06, .12], [.07, .14], [.08, .16]]
      hessians = [[.2, .6], [.3, .9], [.4, 1.2], [.5, 1.5], [.06, .18],
                  [.07, .21], [.08, .24], [.09, .27]]

      # Tests a single feature.
      bucketized_features = [[3], [1], [2], [0], [1], [2], [0], [1]]
      result = boosted_trees_ops.boosted_trees_aggregate_stats(
          node_ids, gradients, hessians, bucketized_features, max_splits,
          num_buckets)
      # shape=[max_splits, num_buckets, feature_dim, stats_dim]
      expected_stats_summary = [
          [[[0., 0., 0., 0.]], [[.08, .16, .09, .27]], [[0., 0., 0., 0.]],
           [[0., 0., 0., 0.]]],
          [[[0., 0., 0., 0.]], [[.15, 0.3, .36, 1.08]], [[.06, 0.12, .07,
                                                          0.21]],
           [[.1, .2, .2, .6]]],
          [[[-.33, -.66, .58, 1.74]], [[0., 0., 0., 0.]], [[.3, .6, .4, 1.2]],
           [[0., 0., 0., 0.]]],
      ]
      expected_stats_summary = np.swapaxes(expected_stats_summary, 1, 2)
      self.assertAllClose(expected_stats_summary, result)

  def _verify_precision(self, length):
    with self.cached_session():
      max_splits = 1
      num_buckets = 1
      node_ids = array_ops.fill([length], 0)

      gradients = constant_op.constant(
          2.0 / length, dtype=dtypes.float32, shape=[length, 1])
      hessians = constant_op.constant(
          0.2 / length, dtype=dtypes.float32, shape=[length, 1])

      bucketized_features = array_ops.zeros([length], dtype=dtypes.int32)

      result = boosted_trees_ops.make_stats_summary(
          node_ids, gradients, hessians, [bucketized_features], max_splits,
          num_buckets)  # shape=[max_splits, num_buckets, num_features, 2]

      self.assertAllClose([[[[2., 0.2]]]], self.evaluate(result))

  def testMakeStatsSummaryNumericalPrecisionSmallBatch(self):
    """Tests numeric precision."""
    self._verify_precision(length=2000)

  def testMakeStatsSummaryNumericalPrecisionMediumBatch(self):
    """Tests numeric precision."""
    self._verify_precision(length=100000)

  def testMakeStatsSummaryNumericalPrecisionLargeBatch(self):
    """Tests numeric precision."""
    self._verify_precision(length=1000000)

  def testMakeStatsSummaryNumericalPrecisionMegaBatch(self):
    """Tests numeric precision."""
    self._verify_precision(length=50000000)


if __name__ == '__main__':
  googletest.main()
