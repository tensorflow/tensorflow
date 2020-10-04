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
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


_INEQUALITY_DEFAULT_LEFT = 'INEQUALITY_DEFAULT_LEFT'.encode('utf-8')
_INEQUALITY_DEFAULT_RIGHT = 'INEQUALITY_DEFAULT_RIGHT'.encode('utf-8')
_EQUALITY_DEFAULT_RIGHT = 'EQUALITY_DEFAULT_RIGHT'.encode('utf-8')


class StatsOpsTest(test_util.TensorFlowTestCase):
  """Tests stats_ops."""

  def _append_zeros_for_default_bucket(self, stats_summary):
    summary_shapes = stats_summary.shape
    # pad zeros for missing value bucket.
    stats_summary = np.concatenate(
        (stats_summary,
         np.zeros([summary_shapes[0], summary_shapes[1], 1, summary_shapes[3]
                  ])),
        axis=2)
    return stats_summary

  def add_f_dim_and_append_zeros(self, stats_summaries):
    """Transform a list of stats summaries, adding a feature dimension.

    The input shape is a list of arrays of shape [max_splits, num_buckets,
    logits+hess dim]. This transformation returns a list of arrays of shape
    [max_splits, 1, num_buckets + 1, logits+hess dim].

    Args:
      stats_summaries: a list of numpy arrays.

    Returns:
      A list of numpy arrays.
    """
    return [
        self._append_zeros_for_default_bucket(np.expand_dims(feature, axis=1))
        for feature in stats_summaries
    ]

  def _get_stats_summary_for_split(self):
    return [
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
    ]  # shape=[feature_dim, max_splits, num_buckets, 2]

  def _get_sparse_stats_summary_for_split(self, stats_summary=None):
    if stats_summary is None:
      stats_summary = np.asarray(self._get_stats_summary_for_split())
      stats_summary[0][0][1] = np.zeros([2])
      stats_summary[1][0][2] = np.zeros([2])
      stats_summary = np.moveaxis(stats_summary, 0, 1)
    slices = stats_summary.nonzero()
    values = stats_summary[slices]
    indices = np.asarray(slices)
    return np.moveaxis(indices, 0, 1), values, stats_summary.shape

  def testCalculateBestSplitsWithoutRegularizationInSparse(self):
    # This test uses the same data as dense, but run in sparse kernel and
    # make sure the sparse kernel returns same result as dense kernel.
    dense_summary = np.asarray([
        [
            [[0., 0.], [.0, .0], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .07], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .58], [0., 0.], [.3, .4]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # feature 0
        [
            [[0., 0.], [0., 0.], [.0, .0], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.3, .5], [-.05, .06], [.06, .07]],  # node 1
            [[.1, .1], [.2, .3], [-.4, .5], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # feature 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    node_id_range = [1, 3]
    dense_summary = np.moveaxis(dense_summary, 0, 1)
    dense_shape = dense_summary.shape

    default_bucket_summary = np.zeros(dense_shape[0:2] + (1, dense_shape[3]))
    sparse_summary = np.concatenate((dense_summary, default_bucket_summary),
                                    axis=2)
    slices = sparse_summary.nonzero()
    summary_values = sparse_summary[slices]
    summary_indices = np.asarray(slices)
    summary_indices = np.moveaxis(summary_indices, 0, 1)
    summary_shape = sparse_summary.shape

    (node_ids, gains, _, _, left_node_contribs, right_node_contribs,
     _) = self.evaluate(
         boosted_trees_ops.sparse_calculate_best_feature_split(
             node_id_range,
             summary_indices,
             summary_values,
             summary_shape,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.02823, 0.41184], gains)
    self.assertAllClose([-0.6], left_node_contribs[0])
    self.assertAllClose([-0.076923], right_node_contribs[0])

  def testSparseCalculateBestSplitsWithoutRegularization(self):
    node_id_range = [1, 3]
    (summary_indices, summary_values,
     summary_shape) = self._get_sparse_stats_summary_for_split()

    (node_ids, gains, feature_dimensions, thresholds, left_node_contribs,
     right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.sparse_calculate_best_feature_split(
             node_id_range,
             summary_indices,
             summary_values,
             summary_shape,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.116495, 0.60429], gains)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllEqual([1, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.631579], [-0.770833]], left_node_contribs)
    self.assertAllClose([[0.833333], [0.8]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestGainsWithoutRegularization_v1_op(self):
    """Testing Gain calculation without any regularization."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = self._get_stats_summary_for_split()

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

  def testCalculateBestFeaturesInvalidSplitType_v2_op(self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [9, 12]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    stats_summaries = self.add_f_dim_and_append_zeros(stats_summaries)

    with self.assertRaisesRegex(Exception, 'Incorrect split type'):
      self.evaluate(
          boosted_trees_ops.calculate_best_feature_split_v2(
              node_id_range,
              stats_summaries,
              split_types=['INVALID'] * len(candidate_feature_ids),
              candidate_feature_ids=candidate_feature_ids,
              l1=0.0,
              l2=0.0,
              tree_complexity=0.0,
              min_node_weight=0,
              logits_dimension=1))

  def testCalculateBestFeaturesWithoutRegularization_v2_op(self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [9, 12]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    stats_summaries = self.add_f_dim_and_append_zeros(stats_summaries)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range,
             stats_summaries,
             split_types=['inequality'] * len(candidate_feature_ids),
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature_id and dimension that has the best gain per node.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.02823, 0.41184], gains)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllEqual([12, 9], feature_ids)
    f_dim = 0  # Both features only have one dimension.
    self.assertAllEqual([f_dim] * 2, feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-.6], [.568966]], left_node_contribs)
    self.assertAllClose([[-.076923], [-.75]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureSplitsWithoutRegularization_v2_op(self):
    """Testing best split without any regularization for a multi-dim feature."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature_id and dimension that has the best gain per node.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.02823, 0.41184], gains)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 0], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-.6], [.568966]], left_node_contribs)
    self.assertAllClose([[-.076923], [-.75]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureSplitWMissingValuesWORegularization_v2_op(
      self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.116495, 0.60429], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.631579], [-0.770833]], left_node_contribs)
    self.assertAllClose([[0.833333], [0.8]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureEqualitySplitsWithoutRegularization_v2_op(
      self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['equality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    # 0.116495 = (-0.05)^2/0.06 + 0.36^2/0.57 - 0.31^2/0.63
    # 0.60429 = (-0.4)^2/0.5 + 0.37^2/0.48 - 0.03^2/0.98
    self.assertAllClose([0.116495, 0.60429], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([2, 2], thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    # left contrib 0.83 = 0.05/0.06, 0.8 = 0.4/0.5
    self.assertAllClose([[0.833333], [.8]], left_node_contribs)
    # right contrib -0.6315 = -0.36/0.57, -0.7708 = -0.37/0.48
    self.assertAllClose([[-0.631579], [-0.770833]], right_node_contribs)
    self.assertAllEqual([_EQUALITY_DEFAULT_RIGHT] * 2, split_types)

  def testCalculateBestMultiDimFeatureMixedSplitTypeWithoutRegularization_v2_op(
      self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [9, 12]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Add in feature dimension.
    stats_summaries = [
        np.expand_dims(feature, axis=1) for feature in stats_summaries
    ]

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range,
             stats_summaries,
             split_types=['inequality', 'equality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    # 0.116495 = (-0.05)^2/0.06 + 0.36^2/0.57 - 0.31^2/0.63
    # 0.60429 = (-0.4)^2/0.5 + 0.37^2/0.48 - 0.03^2/0.98
    self.assertAllClose([0.116495, 0.60429], gains)
    self.assertAllEqual([12, 12], feature_ids)
    f_dim = 0  # Both features only have one dimension.
    self.assertAllEqual([f_dim, f_dim], feature_dimensions)
    self.assertAllEqual([2, 2], thresholds)
    # Same result as equality only test, as feature_1 is chose for both nodes.
    # left contrib 0.83 = 0.05/0.06, 0.8 = 0.4/0.5
    self.assertAllClose([[0.833333], [.8]], left_node_contribs)
    # right contrib -0.6315 = -0.36/0.57, -0.7708 = -0.37/0.48
    self.assertAllClose([[-0.631579], [-0.770833]], right_node_contribs)
    # Feature 1 is inequality.
    self.assertAllEqual([_EQUALITY_DEFAULT_RIGHT, _EQUALITY_DEFAULT_RIGHT],
                        split_types)

  def testCalculateBestGainsWithL2_v1_op(self):
    """Testing Gain calculation with L2."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = self._get_stats_summary_for_split()

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

  def testCalculateMultiDimBestFeatureSplitsWithL2_v2_op(self):
    """Testing best split calculation with L2."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.1,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 0], feature_dimensions)
    self.assertAllClose([0.01879096, 0.33931375], gains)
    self.assertAllEqual([1, 1], thresholds)
    # # The left node contrib will be later added to the previous node value to
    # # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-.5], [.485294]], left_node_contribs)
    self.assertAllClose([[-.043478], [-.6]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateMultiDimBestFeatureSplitsWithMissingValuesL2_v2_op(self):
    """Testing best split calculation with L2."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.1,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllClose([0.077414, 0.501868], gains)
    self.assertAllEqual([1, 1], thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.537313], [-0.637931]], left_node_contribs)
    self.assertAllClose([[0.3125], [0.666667]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateMultiDimBestFeatureEqualitySplitsWithL2_v2_op(self):
    """Testing best split calculation with L2."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['equality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.1,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    # 0.077414 = 0.05^2/0.16 + 0.36^2/0.67 - 0.31^2/0.73
    # 0.501868 = 0.4^2/0.6 + 0.37^2/0.58 - 0.03^2/1.08
    self.assertAllClose([0.077414, 0.501868], gains)
    self.assertAllEqual([2, 2], thresholds)
    # # The left node contrib will be later added to the previous node value to
    # # make the left node value, and the same for right node contrib.
    # left contrib 0.3125 = 0.05/0.16, 0.6667 = 0.4/0.6
    self.assertAllClose([[0.3125], [0.666667]], left_node_contribs)
    # right contrib -0.5373 = -0.36/0.67, -0.6379 = -0.37/0.58
    self.assertAllClose([[-0.537313], [-0.637931]], right_node_contribs)
    self.assertAllEqual([_EQUALITY_DEFAULT_RIGHT] * 2, split_types)

  def testSparseCalculateBestSplitsWithL2(self):
    node_id_range = [1, 3]
    (summary_indices, summary_values,
     summary_shape) = self._get_sparse_stats_summary_for_split()

    (node_ids, gains, feature_dimensions, thresholds, left_node_contribs,
     right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.sparse_calculate_best_feature_split(
             node_id_range,
             summary_indices,
             summary_values,
             summary_shape,
             l1=0.0,
             l2=0.1,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.077414, 0.501868], gains)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.537313], [-0.637931]], left_node_contribs)
    self.assertAllClose([[0.3125], [0.666667]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT, _INEQUALITY_DEFAULT_LEFT],
                        split_types)

  def testCalculateBestGainsWithL1_v1_op(self):
    """Testing Gain calculation with L1."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = self._get_stats_summary_for_split()

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

  def testCalculateBestMultiDimFeatureSplitsWithL1_v2_op(self):
    """Testing best split calculation with L1."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.1,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    # Gain should also include an adjustment of the gradient by l1.
    self.assertAllClose([0.01, 0.191207], gains)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[-0.4], [-0.5]], left_node_contribs)
    self.assertAllClose([[0.], [0.396552]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureSplitsWithMissingValuesL1_v2_op(self):
    """Testing best split calculation with L1."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.1,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    # Gain should also include an adjustment of the gradient by l1.
    # (0.36-0.1)^2/0.57 + 0 - (0.31-0.1)^2/0.63 = 0.048597
    # (0.37-0.1)^2/0.48 + (-0.4+0.1)^2/0.5 = 0.331875
    self.assertAllClose([0.048597, 0.331875], gains)
    self.assertAllEqual([1, 1], thresholds)
    # -(0.36-0.1)/0.57 = -0.45614
    # -(0.37-0.1)/0.48 = -0.5625
    self.assertAllClose([[-0.45614], [-0.5625]], left_node_contribs)
    # -(-0.4+0.1)/0.5 = 0.6
    self.assertAllClose([[0.], [0.6]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureEqualitySplitsWithL1_v2_op(self):
    """Testing best split calculation with L1."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['equality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.1,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    # 0.048597 = 0 + 0.26^2/0.57 - 0.21^2/0.63
    # 0.501868 = 0.3^2/0.5 + 0.27^2/0.48 - 0
    self.assertAllClose([0.048597, 0.331875], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([2, 2], thresholds)
    # # The left node contrib will be later added to the previous node value to
    # # make the left node value, and the same for right node contrib.
    # left contrib 0 (-0.05>-0.1), 0.6 = 0.3/0.5
    self.assertAllClose([[0], [0.6]], left_node_contribs)
    # right contrib -0.45614 = -0.26/0.57, -0.5625 = -0.27/0.48
    self.assertAllClose([[-0.45614], [-0.5625]], right_node_contribs)
    self.assertAllEqual([_EQUALITY_DEFAULT_RIGHT] * 2, split_types)

  def testSparseCalculateBestSplitsWithL1(self):
    node_id_range = [1, 3]
    (summary_indices, summary_values,
     summary_shape) = self._get_sparse_stats_summary_for_split()

    (node_ids, gains, feature_dimensions, thresholds, left_node_contribs,
     right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.sparse_calculate_best_feature_split(
             node_id_range,
             summary_indices,
             summary_values,
             summary_shape,
             l1=0.1,
             l2=0.,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.048597, 0.331875], gains)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.45614], [-0.5625]], left_node_contribs)
    self.assertAllClose([[0.0], [0.6]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestGainsWithTreeComplexity_v1_op(self):
    """Testing best gain calculation with tree complexity."""
    with self.cached_session() as sess:
      max_splits = 7
      node_id_range = [1, 3]  # node 1 through 2 will be processed.
      stats_summary_list = self._get_stats_summary_for_split()

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

  def testCalculateBestMultiDimFeatureSplitsWithTreeComplexity_v2_op(self):
    """Testing best split calculation with tree complexity."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.1,
             tree_complexity=3,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    # Gain should also include an adjustment of the gradient by l1.
    self.assertAllClose([-2.98120904, -2.66068625], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 0], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[-0.5], [0.485294]], left_node_contribs)
    self.assertAllClose([[-0.043478], [-.6]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureSplitsWMissingValsTreeComplexity_v2_op(
      self):
    """Testing best split calculation with tree complexity."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.1,
             tree_complexity=3,
             min_node_weight=0,
             logits_dimension=1))

    # Get same result as v1 op (CalculateBestGainsPerFeature), and find the
    # feature dimension that has the best gain.
    self.assertAllEqual([1, 2], node_ids)
    # Gain should also include an adjustment of the gradient by l1.
    self.assertAllClose([-2.922586, -2.498132], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[-0.537313], [-0.637931]], left_node_contribs)
    self.assertAllClose([[0.3125], [0.666667]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestMultiDimFeatureEqualitySplitsWithTreeComplexity_v2_op(
      self):
    """Testing best split calculation with tree complexity."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summaries = self._get_stats_summary_for_split()
    # Convert from list of arrays to a single array and reshape to [max_splits,
    # feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summaries, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['equality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.1,
             tree_complexity=3,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    # -2.922586 = 0.05^2/0.16 + 0.36^2/0.67 - 0.31^2/0.73 - 3
    # -2.498132 = 0.4^2/0.6 + 0.37^2/0.58 - 0.03^2/1.08 - 3
    self.assertAllClose([-2.922586, -2.498132], gains)
    self.assertAllEqual([2, 2], thresholds)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    # # The left node contrib will be later added to the previous node value to
    # # make the left node value, and the same for right node contrib.
    # left contrib 0.3125 = 0.05/0.16, 0.6667 = 0.4/0.6
    self.assertAllClose([[0.3125], [0.666667]], left_node_contribs)
    # right contrib -0.5373 = -0.36/0.67, -0.6379 = -0.37/0.58
    self.assertAllClose([[-0.537313], [-0.637931]], right_node_contribs)
    self.assertAllEqual([_EQUALITY_DEFAULT_RIGHT] * 2, split_types)

  def testSparseCalculateBestSplitsWithTreeComplexity(self):
    """Testing best split calculation with tree complexity."""
    node_id_range = [1, 3]
    (summary_indices, summary_values,
     summary_shape) = self._get_sparse_stats_summary_for_split()

    (node_ids, gains, feature_dimensions, thresholds, left_node_contribs,
     right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.sparse_calculate_best_feature_split(
             node_id_range,
             summary_indices,
             summary_values,
             summary_shape,
             l1=0.,
             l2=0.1,
             tree_complexity=3.,
             min_node_weight=0,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([-2.922586, -2.498132], gains)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[-0.537313], [-0.637931]], left_node_contribs)
    self.assertAllClose([[0.3125], [0.666667]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestGainsWithMinNodeWeight_v1_op(self):
    """Testing Gain calculation with min node weight."""
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
      ]  # feature_dim * shape=[max_splits, num_buckets, 2]

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

  def testCalculateMultiDimBestSplitsWithMinNodeWeight_v2_op(self):
    """Testing best split calculation with min node weight."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summary = np.asarray([
        [
            [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .61], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .68], [0., 0.], [.3, .4]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 0
        [
            [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.3, .5], [-.05, .6], [.06, .07]],  # node 1
            [[.1, 1.], [.2, -.05], [-.4, .05], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    # Reshape to [max_splits, feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summary, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=1,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    # Gain should also include an adjustment of the gradient by l1.
    self.assertAllClose([0.098013, 0.931596], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[-.6], [-0.315789]], left_node_contribs)
    self.assertAllClose([[-0.014925], [2.53846]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateMultiDimBestSplitsWithMissingValuesMinNodeWeight_v2_op(self):
    """Testing best split calculation with min node weight."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summary = np.asarray([
        [
            [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .61], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .68], [0., 0.], [.3, .4]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 0
        [
            [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.3, .5], [-.05, .6], [.06, .07]],  # node 1
            [[.1, 1.], [.2, -.05], [-.4, .05], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    # Reshape to [max_splits, feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summary, 0, 1)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=1,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    # Gain should also include an adjustment of the gradient by l1.
    self.assertAllClose([0.149398, 3.332075], gains)
    self.assertAllEqual([4, 4], feature_ids)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[-0.631579], [-0.359223]], left_node_contribs)
    self.assertAllClose([[0.083333], [7.999989]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testSparseCalculateBestSplitsWithMinNodeWeight(self):
    """Testing best split calculation with min node weight."""
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summary = np.asarray([
        [
            [[0., 0.], [.0, .0], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .61], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .68], [0., 0.], [.3, .4]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # feature 0
        [
            [[0., 0.], [0., 0.], [.0, .0], [0., 0.]],  # node 0; ignored
            [[0., 0.], [-.05, .6], [.3, .5], [.06, .07]],  # node 1
            [[.1, 1.], [.2, -.05], [-.4, .05], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # feature 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    # reshape to [max_splits, feature_dim, num_buckets, 2]
    stats_summary = np.moveaxis(stats_summary, 0, 1)

    (summary_indices, summary_values,
     summary_shape) = self._get_sparse_stats_summary_for_split(stats_summary)

    (node_ids, gains, feature_dimensions, thresholds, left_node_contribs,
     right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.sparse_calculate_best_feature_split(
             node_id_range,
             summary_indices,
             summary_values,
             summary_shape,
             l1=0.,
             l2=0.,
             tree_complexity=0.,
             min_node_weight=1,
             logits_dimension=1))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.149398, 3.332079], gains)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllClose([[0.083333], [-0.359223]], left_node_contribs)
    self.assertAllClose([[-0.631579], [7.999998]], right_node_contribs)
    self.assertAllEqual([1, 1], feature_dimensions)
    self.assertAllEqual([_INEQUALITY_DEFAULT_RIGHT, _INEQUALITY_DEFAULT_LEFT],
                        split_types)

  def testCalculateBestGainsWithMinNodeWeightNoSplitOnFeaturePossible_v1_op(
      self):
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
      ]  # feature_dim * shape=[max_splits, num_buckets, 2]

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

  def testCalculateBestMultiDimFeatureSplitsWithNoSplitOnFeaturePossible_v2_op(
      self):
    """Testing best split calculation with min node weight and no split."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summary = np.asarray([
        [
            [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .7], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .068], [0., 0.], [.3, .04]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 0
        [
            [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.3, .5], [-.05, .06], [.06, .7]],  # node 1
            [[.1, .1], [.2, -.05], [-.4, .05], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    # Reshape to [max_splits, feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summary, 0, 1)
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, _, _, _, _, _, _,
     _) = boosted_trees_ops.calculate_best_feature_split_v2(
         node_id_range, [stats_summary],
         split_types=['inequality'],
         candidate_feature_ids=candidate_feature_ids,
         l1=0.0,
         l2=0.0,
         tree_complexity=0.0,
         min_node_weight=1,
         logits_dimension=1)

    # We can't split either of the nodes on the first feature.
    self.assertAllEqual([1], node_ids)

    # Now check when we can't split on any feature.
    (node_ids, _, _, _, _, _, _,
     _) = boosted_trees_ops.calculate_best_feature_split_v2(
         node_id_range, [stats_summary],
         split_types=['inequality'],
         candidate_feature_ids=candidate_feature_ids,
         l1=0.0,
         l2=0.0,
         tree_complexity=0.0,
         min_node_weight=10,
         logits_dimension=1)
    self.assertAllEqual([], node_ids)

  def testCalculateBestMultiDimFeatureEqualitySplitsWithNoSplitPossible_v2_op(
      self):
    """Testing best split calculation with min node weight and no split."""
    candidate_feature_ids = [4]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summary = np.asarray([
        [
            [[0., 0.], [.08, .09], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .7], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .068], [0., 0.], [.3, .04]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 0
        [
            [[0., 0.], [0., 0.], [.08, .09], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.3, .5], [-.05, .06], [.06, .7]],  # node 1
            [[.1, .1], [.2, -.05], [-.4, .05], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # f_dim 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    # Reshape to [max_splits, feature_dim, num_buckets, 2].
    stats_summary = np.moveaxis(stats_summary, 0, 1)

    (node_ids, _, _, _, _, _, _,
     _) = boosted_trees_ops.calculate_best_feature_split_v2(
         node_id_range, [stats_summary],
         split_types=['equality'],
         candidate_feature_ids=candidate_feature_ids,
         l1=0.0,
         l2=0.0,
         tree_complexity=0.0,
         min_node_weight=1,
         logits_dimension=1)

    # We can't split either of the nodes on the first feature
    self.assertAllEqual([1], node_ids)

    # Now check when we can't split on any feature
    (node_ids, _, _, _, _, _, _,
     _) = boosted_trees_ops.calculate_best_feature_split_v2(
         node_id_range, [stats_summary],
         split_types=['equality'],
         candidate_feature_ids=candidate_feature_ids,
         l1=0.0,
         l2=0.0,
         tree_complexity=0.0,
         min_node_weight=10,
         logits_dimension=1)
    self.assertAllEqual([], node_ids)

  def testSparseCalculateBestSplitsWithMinNodeWeightNoSplitOnFeature(self):
    """Testing best split calculation with min node weight and no split."""
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    stats_summary = np.asarray([
        [
            [[0., 0.], [.0, .0], [0., 0.], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.15, .36], [.06, .7], [.1, .2]],  # node 1
            [[0., 0.], [-.33, .068], [0., 0.], [.3, .04]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # feature 0
        [
            [[0., 0.], [0., 0.], [.0, .0], [0., 0.]],  # node 0; ignored
            [[0., 0.], [.3, .5], [-.05, .6], [.06, .07]],  # node 1
            [[.1, .1], [.2, .03], [-.4, .05], [.07, .08]],  # node 2
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 3; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 4; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 5; ignored
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],  # node 6; ignored
        ],  # feature 1
    ])  # feature_dim * shape=[max_splits, num_buckets, 2]
    # reshape to [max_splits, feature_dim, num_buckets, 2]
    stats_summary = np.moveaxis(stats_summary, 0, 1)
    (summary_indices, summary_values,
     summary_shape) = self._get_sparse_stats_summary_for_split(stats_summary)

    (node_ids, _, _, _, _, _, _) = self.evaluate(
        boosted_trees_ops.sparse_calculate_best_feature_split(
            node_id_range,
            summary_indices,
            summary_values,
            summary_shape,
            l1=0.,
            l2=0.,
            tree_complexity=0.,
            min_node_weight=1,
            logits_dimension=1))

    # We can't split either of the nodes on the first feature
    self.assertAllEqual([1], node_ids)

    # Now check when we can't split on any feature
    (node_ids, _, _, _, _, _, _) = self.evaluate(
        boosted_trees_ops.sparse_calculate_best_feature_split(
            node_id_range,
            summary_indices,
            summary_values,
            summary_shape,
            l1=0.,
            l2=0.,
            tree_complexity=0.,
            min_node_weight=10,
            logits_dimension=1))
    self.assertAllEqual([], node_ids)

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
    expected_stats_summary = np.asarray(
        [1., 5., 2., 6., 0., 0., 3., 7., 4., 8., 0., 0.])
    # shape=[max_splits, num_buckets, feature_dim, stats_dim]
    expected_stats_summary = np.reshape(expected_stats_summary, (2, 3, 1, 2))
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
          num_buckets)  # shape=[max_splits, num_buckets, feature_dim, 2]
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
        [[[0., 0.]], [[.08, .09]], [[0., 0.]], [[0., 0.]], [[0., 0.]]],
        [[[0., 0.]], [[.15, .36]], [[.06, .07]], [[.1, .2]], [[0., 0.]]],
        [[[-.33, .58]], [[0., 0.]], [[.3, .4]], [[0., 0.]], [[0., 0.]]],
    ]
    # Swap feature dim and bucket id axis
    expected_stats_summary = np.swapaxes(expected_stats_summary, 1, 2)
    self.assertAllClose(expected_stats_summary, result)

  def testAggregateStatsAccumulateWithMissingValue(self):
    """Tests that Summary actually accumulates."""
    max_splits = 3
    num_buckets = 4
    node_ids = [1, 1, 2, 2, 1, 1, 2, 0]
    gradients = [[.1], [.2], [.3], [-.4], [-.05], [.06], [.07], [.08]]
    hessians = [[.2], [.3], [.4], [.5], [.06], [.07], [.08], [.09]]

    # Tests a single feature.
    missing_feature = -1
    bucketized_features = [[3], [1], [2], [0], [missing_feature], [2], [0], [1]]
    result = boosted_trees_ops.boosted_trees_aggregate_stats(
        node_ids, gradients, hessians, bucketized_features, max_splits,
        num_buckets)
    # shape=[max_splits, num_buckets, feature_dim, stats_dim]
    # Get the same result as MakeStatsSummary Op.
    expected_stats_summary = [
        [[[0., 0.]], [[.08, .09]], [[0., 0.]], [[0., 0.]], [[0., 0.]]],
        [[[0., 0.]], [[.2, .3]], [[.06, .07]], [[.1, .2]], [[-.05, .06]]],
        [[[-.33, .58]], [[0., 0.]], [[.3, .4]], [[0., 0.]], [[0., 0.]]],
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
          num_buckets)  # shape=[max_splits, num_buckets, feature_dim, 2]
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
        [[0, 0, 0, 0, .08, .09, 0, 0, 0, 0, .08, .09, 0, 0, 0, 0, 0, 0, 0, 0],
         [
             0, 0, .3, .5, .15, .36, 0, 0, .06, .07, -.05, .06, .1, .2, .06,
             .07, 0, 0, 0, 0
         ],
         [
             -.33, .58, .3, .4, 0, 0, 0, 0, .3, .4, -.4, .5, 0, 0, .07, .08, 0,
             0, 0, 0
         ]])
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
      expected_stats_summary = np.reshape(expected_stats_summary, (3, 5, 2, 2))
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
           [[0., 0., 0., 0.]], [[0., 0., 0., 0.]]],
          [[[0., 0., 0., 0.]], [[.15, 0.3, .36, 1.08]], [[.06, 0.12, .07,
                                                          0.21]],
           [[.1, .2, .2, .6]], [[0., 0., 0., 0.]]],
          [[[-.33, -.66, .58, 1.74]], [[0., 0., 0., 0.]], [[.3, .6, .4, 1.2]],
           [[0., 0., 0., 0.]], [[0., 0., 0., 0.]]],
      ]
      expected_stats_summary = np.swapaxes(expected_stats_summary, 1, 2)
      self.assertAllClose(expected_stats_summary, result)

  def _get_dense_summaries_from_sparse_features(self, max_splits, num_buckets,
                                                batch_size, feature_dims,
                                                logits_dims, hess_dims):
    np.random.seed(0)
    stats_dims = logits_dims + hess_dims
    node_ids = np.random.randint(max_splits, size=batch_size)
    gradients = np.random.uniform(5.0, size=(batch_size, logits_dims))
    hessians = np.random.uniform(5.0, size=(batch_size, hess_dims))
    dense_indices = np.random.randint(2, size=(batch_size, feature_dims))
    feature_indices = np.argwhere(dense_indices == 1)
    missing_feature_indices = np.argwhere(dense_indices == 0)
    feature_values = np.random.randint(num_buckets, size=len(feature_indices))
    feature_shape = np.asarray([batch_size, feature_dims])
    # Last bucket is for missing values.
    dense_summary = np.zeros(
        (max_splits, feature_dims, num_buckets + 1, stats_dims))
    for (instance, f_dim), bucket in zip(feature_indices, feature_values):
      node_id = node_ids[instance]
      dense_summary[node_id][f_dim][bucket] += np.concatenate(
          [gradients[instance], hessians[instance]])

    for instance, f_dim in missing_feature_indices:
      node_id = node_ids[instance]
      dense_summary[node_id][f_dim][num_buckets] += np.concatenate(
          [gradients[instance], hessians[instance]])

    return (node_ids, gradients, hessians, feature_indices, feature_values,
            feature_shape, dense_summary)

  def testMakeSparseStatsSummarySingleFeatureDimension(self):
    batch_size = 10
    max_splits = 2
    num_buckets = 2
    feature_dims = 1
    logits_dims = 1
    hess_dims = 1

    (node_ids, gradients, hessians, feature_indices, feature_values,
     feature_shape,
     expected_dense_summary) = self._get_dense_summaries_from_sparse_features(
         max_splits, num_buckets, batch_size, feature_dims, logits_dims,
         hess_dims)

    (summary_indices, summary_values,
     summary_shape) = boosted_trees_ops.boosted_trees_sparse_aggregate_stats(
         node_ids, gradients, hessians, feature_indices, feature_values,
         feature_shape, max_splits, num_buckets)
    dense_result = sparse_ops.sparse_to_dense(summary_indices, summary_shape,
                                              summary_values)
    self.assertAllClose(expected_dense_summary, dense_result)

  def testMakeSparseStatsSummaryMultiDimFeature(self):
    batch_size = 10
    max_splits = 2
    num_buckets = 2
    feature_dims = 1
    logits_dims = 1
    hess_dims = 1

    (node_ids, gradients, hessians, feature_indices, feature_values,
     feature_shape,
     expected_dense_summary) = self._get_dense_summaries_from_sparse_features(
         max_splits, num_buckets, batch_size, feature_dims, logits_dims,
         hess_dims)

    (summary_indices, summary_values,
     summary_shape) = boosted_trees_ops.boosted_trees_sparse_aggregate_stats(
         node_ids, gradients, hessians, feature_indices, feature_values,
         feature_shape, max_splits, num_buckets)
    dense_result = sparse_ops.sparse_to_dense(summary_indices, summary_shape,
                                              summary_values)
    self.assertAllClose(expected_dense_summary, dense_result)

  def testMakeSparseStatsSummaryMultiClass(self):
    batch_size = 10
    max_splits = 2
    num_buckets = 2
    feature_dims = 1
    logits_dims = 2
    hess_dims = 2

    (node_ids, gradients, hessians, feature_indices, feature_values,
     feature_shape,
     expected_dense_summary) = self._get_dense_summaries_from_sparse_features(
         max_splits, num_buckets, batch_size, feature_dims, logits_dims,
         hess_dims)

    (summary_indices, summary_values,
     summary_shape) = boosted_trees_ops.boosted_trees_sparse_aggregate_stats(
         node_ids, gradients, hessians, feature_indices, feature_values,
         feature_shape, max_splits, num_buckets)
    dense_result = sparse_ops.sparse_to_dense(summary_indices, summary_shape,
                                              summary_values)
    self.assertAllClose(expected_dense_summary, dense_result)

  def testMakeSparseStatsSummaryMultiClassAndMultiFeatureDim(self):
    batch_size = 10
    max_splits = 2
    num_buckets = 2
    feature_dim = 2
    logits_dims = 2
    hess_dims = 2

    (node_ids, gradients, hessians, feature_indices, feature_values,
     feature_shape,
     expected_dense_summary) = self._get_dense_summaries_from_sparse_features(
         max_splits, num_buckets, batch_size, feature_dim, logits_dims,
         hess_dims)

    (summary_indices, summary_values,
     summary_shape) = boosted_trees_ops.boosted_trees_sparse_aggregate_stats(
         node_ids, gradients, hessians, feature_indices, feature_values,
         feature_shape, max_splits, num_buckets)
    dense_result = sparse_ops.sparse_to_dense(summary_indices, summary_shape,
                                              summary_values)
    self.assertAllClose(expected_dense_summary, dense_result)

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
          num_buckets)  # shape=[max_splits, num_buckets, feature_dim, 2]

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


class BestMultiDimFeatureSplitMultiClassV2Op(StatsOpsTest):
  """Tests multi-class/multi-regression for best splits using V2 op."""

  logits_dim = 2

  def _get_stats_summary_for_split_diagonal_hessian(self):
    summary = [
        [[[0., 0., 0., 0.], [0.08, 0.2, 0.09, 0.3], [0., 0., 0., 0.],
          [0., 0., 0., 0.]],
         [[0., 0., 0., 0.], [0., 0., 0., 0.], [0.08, 0.2, 0.09, 0.3],
          [0., 0., 0., 0.]]],  # node 0
        [[[0., 0., 0., 0.], [-0.25, -0.1, 0.36, 0.2], [-0.14, 0.25, 0.07, 0.18],
          [0.1, 0.235, 0.2, 0.06]],
         [[0., 0., 0., 0.], [-0.3, 0.12, 0.5, 0.31], [-0.05, 0.115, 0.11, 0.09],
          [0.06, 0.15, 0.02, 0.04]]],  # node 1
        [[[0., 0., 0., 0.], [-0.03, 0.21, 0.28, 0.44], [0., 0., 0., 0.],
          [0.3, 0.04, 0.4, 0.41]],
         [[0.4, 0.188, 0.16, -0.03], [0.2, -0.088, 0.1, -0.24],
          [-0.4, -0.06, 0.5, 0.15], [0.07, 0.21, -0.08, 0.97]]],  # node 2
        [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]],
         [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]]],  # node 3
        [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]],
         [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]]],  # node 4
        [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]],
         [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]]],  # node 5
        [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]],
         [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
          [0., 0., 0., 0.]]]  # node 6
    ]
    # [max_splits, feature_dim, num_buckets, 4]
    return np.array(summary)

  def _add_feature_dim(self, stats_summary):
    """Add dimension for features; number of features will be 1."""
    return np.expand_dims(stats_summary, axis=1)

  def testSumOfStatsSummaryValuesFromHelperFunction(self):
    """Sum of grads and hessians is correct from helper function."""
    # [max_splits, feature_dim, num_buckets, 4]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()
    # Test that sum of grads/hessians are same for both features for all nodes.
    # [max_splits, feature_dim, 4]
    agg = stats_summary.sum(axis=2)  # Sum along buckets.
    self.assertAllClose(agg[:, 0, :], agg[:, 1, :])  # There are two features.
    # Test sum of hessians for each nodes. These values are used to evaluate if
    # node meets min_node_weight criteria.
    nodes_agg = agg[:, 0, :]
    hessians = nodes_agg[:, self.logits_dim:]

    def frobenius(x, **kwargs):
      return np.sqrt(np.square(x).sum(**kwargs))

    self.assertAllClose([0.3132092, 0.76843998, 1.08853112, 0., 0., 0., 0.],
                        frobenius(hessians, axis=1))

  def testCalculateBestFeatureSplitsSingleClassVsMultiClass(self):
    """Testing same results using same grads/hess with both single and multi."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.

    # Build same stats summary in single class and multi-class form (using
    # diagonal hessian).
    empty = [0] * 2
    stats_summary = [
        [empty, [.08, .09], empty],  # node 0; ignored
        [empty, [-0.25, 0.11], [0.1, 0.5]],  # node 1
        [empty, [0.14, 0.1], empty],  # node 2
        [empty, empty, empty],  # node 3; ignored
    ]
    # [max_splits, feature_dim, num_buckets, 2]
    stats_summary = self._add_feature_dim(stats_summary)
    diag_empty = [0] * 4
    diag_stats_summary = [
        [diag_empty, [0, .08, 0, 0.09], diag_empty],  # node 0; ignored
        [diag_empty, [0, -0.25, 0, 0.11], [0, 0.1, 0, 0.5]],  # node 1
        [diag_empty, [0, 0.14, 0, 0.1], diag_empty],  # node 2
        [diag_empty, diag_empty, diag_empty],  # node 3; ignored
    ]
    # [max_splits, feature_dim, num_buckets, 4]
    diag_stats_summary = self._add_feature_dim(diag_stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=1))

    (diag_node_ids, diag_gains, diag_feature_ids, diag_feature_dimensions,
     diag_thresholds, diag_left_node_contribs, diag_right_node_contribs,
     diag_split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [diag_stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=2))

    self.assertAllEqual(node_ids, diag_node_ids)
    self.assertAllClose(gains, diag_gains)
    self.assertAllEqual(feature_ids, diag_feature_ids)
    self.assertAllEqual(feature_dimensions, diag_feature_dimensions)
    self.assertAllEqual(thresholds, diag_thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    zeros = np.zeros_like(left_node_contribs)
    self.assertAllClose(
        np.concatenate([zeros, left_node_contribs], axis=1),
        diag_left_node_contribs)
    self.assertAllClose(
        np.concatenate([zeros, right_node_contribs], axis=1),
        diag_right_node_contribs)
    self.assertAllEqual(split_types, diag_split_types)

  def testCalculateBestFeatureSplitsDiagonalVsFull(self):
    """Test results are same using diagonal hessian and full hessian."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.

    # Build same stats summary in diagonal and full hessian form, respectively.
    diag_empty = [0] * 4
    diag_stats_summary = [
        [diag_empty, [.08, .09, -.1, .2], diag_empty],  # node 0; ignored
        [diag_empty, [.15, .36, .21, -.11], [.06, .07, .67, 0.5]],  # node 1
        [diag_empty, [-.33, .58, -.2, -.31], diag_empty],  # node 2
        [diag_empty, diag_empty, diag_empty],  # node 3; ignored
    ]
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    diag_stats_summary = self._add_feature_dim(diag_stats_summary)
    full_empty = [0] * 6
    full_stats_summary = [
        [full_empty, [.08, .09, -.1, 0, 0, .2], full_empty],  # node 0; ignored
        [full_empty, [.15, .36, .21, 0, 0, -.11], [.06, .07, .67, 0, 0,
                                                   0.5]],  # node 1
        [full_empty, [-.33, .58, -.2, 0, 0, -.31], full_empty],  # node 2
        [full_empty, full_empty, full_empty],  # node 3; ignored
    ]
    # [max_splits, feature_dim, num_buckets, logits_dim + logits_dim**2]
    full_stats_summary = self._add_feature_dim(full_stats_summary)
    (diag_node_ids, diag_gains, diag_feature_ids, diag_feature_dimensions,
     diag_thresholds, diag_left_node_contribs, diag_right_node_contribs,
     diag_split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [diag_stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    (full_node_ids, full_gains, full_feature_ids, full_feature_dimensions,
     full_thresholds, full_left_node_contribs, full_right_node_contribs,
     full_split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [full_stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    self.assertAllEqual(diag_node_ids, full_node_ids)
    self.assertAllClose(diag_gains, full_gains)
    self.assertAllEqual(diag_feature_ids, full_feature_ids)
    self.assertAllEqual(diag_feature_dimensions, full_feature_dimensions)
    self.assertAllEqual(diag_thresholds, full_thresholds)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose(diag_left_node_contribs, full_left_node_contribs)
    self.assertAllClose(diag_right_node_contribs, full_right_node_contribs)
    self.assertAllEqual(diag_split_types, full_split_types)

  def testCalculateBestFeatureSplitsWithoutRegularization(self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.912981, 1.446218], gains)
    self.assertAllEqual([2, 1], thresholds)
    self.assertAllEqual([14, 14], feature_ids)
    self.assertAllEqual([0, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[0.906977, -0.394737], [-2.307692, 0.370370]],
                        left_node_contribs)
    self.assertAllClose([[-0.5, -3.916667], [0.785714, -0.133928]],
                        right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestFeatureSplitsWMissingValuesWoRegularization(self):
    """Testing best split calculation without any regularization."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.912981, 2.79444], gains)
    self.assertAllEqual([0, 1], thresholds)
    self.assertAllEqual([14, 14], feature_ids)
    self.assertAllEqual([0, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.5, -3.916667], [-3.722223, -0.442857]],
                        left_node_contribs)
    self.assertAllClose([[0.906977, -0.394737], [0.8, 0.4]],
                        right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestFeatureSplitsWithL2(self):
    """Testing best split calculation inith L2 regularization."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()
    stats_summary = self._append_zeros_for_default_bucket(stats_summary)

    l2 = 0.1
    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=l2,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.475669, 1.009791], gains)
    self.assertAllEqual([1, 1], thresholds)
    self.assertAllEqual([14, 14], feature_ids)
    self.assertAllEqual([0, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[0.543478, 0.333333], [-1.666667, 0.588235]],
                        left_node_contribs)
    self.assertAllClose([[0.108108, -1.426471], [0.634615, -0.122951]],
                        right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestFeatureSplitsWithMissingValuesL2(self):
    """Testing best split calculation inith L2 regularization."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()

    l2 = 0.1
    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=l2,
             tree_complexity=0.0,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.475669, 3.467833], gains)
    self.assertAllEqual([1, 0], thresholds)
    self.assertAllEqual([14, 14], feature_ids)
    self.assertAllEqual([0, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[0.543478, 0.333333], [-2.611111, -0.382692]],
                        left_node_contribs)
    self.assertAllClose([[0.108108, -1.426471], [0.285714, 14.800049]],
                        right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_RIGHT, _INEQUALITY_DEFAULT_LEFT],
                        split_types)

  def testCalculateBestFeatureSplitsWithMinNodeWeight(self):
    """Testing best split calculation with min_node_weight."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()

    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=0.5,
             logits_dimension=self.logits_dim))

    # Both nodes have large enough sum(hessians) so use them.
    self.assertAllEqual([1, 2], node_ids)
    self.assertAllClose([0.912981, 2.79444], gains)
    self.assertAllEqual([0, 1], thresholds)
    self.assertAllEqual([14, 14], feature_ids)
    self.assertAllEqual([0, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-0.5, -3.916667], [-3.722223, -0.442857]],
                        left_node_contribs)
    self.assertAllClose([[0.906977, -0.394737], [0.8, 0.4]],
                        right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT] * 2, split_types)

  def testCalculateBestFeatureSplitsWithTreeComplexity(self):
    """Testing best split calculation with tree complexity."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()

    l2 = 0.1
    tree_complexity = 3.
    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=l2,
             tree_complexity=tree_complexity,
             min_node_weight=0,
             logits_dimension=self.logits_dim))

    self.assertAllEqual([1, 2], node_ids)
    self.assertAllEqual([1, 2], node_ids)
    # L2 test result, but subtracted by tree_complexity.
    self.assertAllClose([-2.524331, 0.467833], gains)
    self.assertAllEqual([1, 0], thresholds)
    self.assertAllEqual([14, 14], feature_ids)
    self.assertAllEqual([0, 1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[0.543478, 0.333333], [-2.611111, -0.382692]],
                        left_node_contribs)
    self.assertAllClose([[0.108108, -1.426471], [0.285714, 14.800049]],
                        right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_RIGHT, _INEQUALITY_DEFAULT_LEFT],
                        split_types)

  def testCalculateBestFeatureSplitsWithMinNodeNoSplitOnFeaturePossible(self):
    """Test when parent node hessian doesn't meet min node weight."""
    candidate_feature_ids = [14]
    node_id_range = [1, 3]  # node 1 through 2 will be processed.
    # [max_splits, feature_dim, num_buckets, 2*logits_dim]
    stats_summary = self._get_stats_summary_for_split_diagonal_hessian()

    min_node_weight = 0.8
    (node_ids, gains, feature_ids, feature_dimensions, thresholds,
     left_node_contribs, right_node_contribs, split_types) = self.evaluate(
         boosted_trees_ops.calculate_best_feature_split_v2(
             node_id_range, [stats_summary],
             split_types=['inequality'],
             candidate_feature_ids=candidate_feature_ids,
             l1=0.0,
             l2=0.0,
             tree_complexity=0.0,
             min_node_weight=min_node_weight,
             logits_dimension=self.logits_dim))

    # node_1 doesn't have large enough sum(hessians) so don't return it.
    self.assertAllEqual([2], node_ids)
    self.assertAllClose([2.79444], gains)
    self.assertAllEqual([1], thresholds)
    self.assertAllEqual([14], feature_ids)
    self.assertAllEqual([1], feature_dimensions)
    # The left node contrib will be later added to the previous node value to
    # make the left node value, and the same for right node contrib.
    self.assertAllClose([[-3.722223, -0.442857]], left_node_contribs)
    self.assertAllClose([[0.8, 0.4]], right_node_contribs)
    self.assertAllEqual([_INEQUALITY_DEFAULT_LEFT], split_types)


if __name__ == '__main__':
  googletest.main()
