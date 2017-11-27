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
"""Tests for the GTFlow split handler Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.proto import split_info_pb2
from tensorflow.contrib.boosted_trees.python.ops import split_handler_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class SplitHandlerOpsTest(test_util.TensorFlowTestCase):

  def testMakeDenseSplit(self):
    """Tests split handler op."""
    with self.test_session() as sess:
      # The data looks like the following after dividing by number of steps (2).
      # Gradients    | Partition | Dense Quantile |
      # (1.2, 0.2)   | 0         | 0              |
      # (-0.3, 0.19) | 0         | 1              |
      # (4.0, 0.13)  | 1         | 1              |
      partition_ids = array_ops.constant([0, 0, 1], dtype=dtypes.int32)
      bucket_ids = array_ops.constant(
          [[0, 0], [1, 0], [1, 0]], dtype=dtypes.int64)
      gradients = array_ops.constant([2.4, -0.6, 8.0])
      hessians = array_ops.constant([0.4, 0.38, 0.26])
      bucket_boundaries = [0.3, 0.52]
      partitions, gains, splits = (
          split_handler_ops.build_dense_inequality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              bucket_ids=bucket_ids,
              gradients=gradients,
              hessians=hessians,
              bucket_boundaries=bucket_boundaries,
              l1_regularization=0.1,
              l2_regularization=1,
              tree_complexity_regularization=0,
              min_node_weight=0,
              class_id=-1,
              feature_column_group_id=0,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = sess.run([partitions, gains, splits])
    self.assertAllEqual([0, 1], partitions)

    # Check the split on partition 0.
    # -(1.2 - 0.1) / (0.2 + 1)
    expected_left_weight = -0.91666

    # expected_left_weight * -(1.2 - 0.1)
    expected_left_gain = 1.0083333333333331

    # (-0.3 + 0.1) / (0.19 + 1)
    expected_right_weight = 0.1680672

    # expected_right_weight * -(-0.3 + 0.1)
    expected_right_gain = 0.033613445378151252

    # (-0.3 + 1.2 - 0.1) ** 2 / (0.19 + 0.2 + 1)
    expected_bias_gain = 0.46043165467625885

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    self.assertAllClose(
        expected_left_gain + expected_right_gain - expected_bias_gain, gains[0],
        0.00001)
    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)
    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)
    self.assertEqual(0, split_node.feature_column)
    self.assertAllClose(0.3, split_node.threshold, 0.00001)

    # Check the split on partition 1.
    # (-4 + 0.1) / (0.13 + 1)
    expected_left_weight = -3.4513274336283186
    expected_right_weight = 0
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    # There's only one active bucket here so zero gain is expected.
    self.assertAllClose(0.0, gains[1], 0.00001)
    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)
    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)
    self.assertEqual(0, split_node.feature_column)
    self.assertAllClose(0.52, split_node.threshold, 0.00001)

  def testMakeMulticlassDenseSplit(self):
    """Tests split handler op."""
    with self.test_session() as sess:
      partition_ids = array_ops.constant([0, 0, 1], dtype=dtypes.int32)
      bucket_ids = array_ops.constant(
          [[0, 0], [1, 0], [1, 0]], dtype=dtypes.int64)
      gradients = array_ops.constant([[2.4, 3.0], [-0.6, 0.1], [8.0, 1.0]])
      hessians = array_ops.constant([[[0.4, 1], [1, 1]], [[0.38, 1], [1, 1]],
                                     [[0.26, 1], [1, 1]]])
      bucket_boundaries = [0.3, 0.52]
      partitions, gains, splits = (
          split_handler_ops.build_dense_inequality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              bucket_ids=bucket_ids,
              gradients=gradients,
              hessians=hessians,
              bucket_boundaries=bucket_boundaries,
              l1_regularization=0,
              l2_regularization=1,
              tree_complexity_regularization=0,
              min_node_weight=0,
              class_id=-1,
              feature_column_group_id=0,
              multiclass_strategy=learner_pb2.LearnerConfig.FULL_HESSIAN))
      partitions, gains, splits = sess.run([partitions, gains, splits])
    self.assertAllEqual([0, 1], partitions)

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])

    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split

    # Each leaf has 2 element vector.
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(2, len(right_child.value))
    self.assertEqual(0, split_node.feature_column)
    self.assertAllClose(0.3, split_node.threshold, 1e-6)

  def testMakeDenseSplitEmptyInputs(self):
    """Tests empty inputs op."""
    with self.test_session() as sess:
      partition_ids = array_ops.constant([], dtype=dtypes.int32)
      bucket_ids = array_ops.constant([[]], dtype=dtypes.int64)
      gradients = array_ops.constant([])
      hessians = array_ops.constant([])
      bucket_boundaries = [0.3, 0.52]
      partitions, gains, splits = (
          split_handler_ops.build_dense_inequality_splits(
              num_minibatches=0,
              partition_ids=partition_ids,
              bucket_ids=bucket_ids,
              gradients=gradients,
              hessians=hessians,
              bucket_boundaries=bucket_boundaries,
              l1_regularization=0.1,
              l2_regularization=1,
              tree_complexity_regularization=0,
              min_node_weight=0,
              class_id=-1,
              feature_column_group_id=0,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = sess.run([partitions, gains, splits])
    # .assertEmpty doesn't exist on ubuntu-contrib
    self.assertEqual(0, len(partitions))
    self.assertEqual(0, len(gains))
    self.assertEqual(0, len(splits))

  def testMakeSparseSplit(self):
    """Tests split handler op."""
    with self.test_session() as sess:
      # The data looks like the following after dividing by number of steps (2).
      # Gradients    | Partition | bucket ID       |
      # (0.9, 0.39)  | 0         | -1              |
      # (1.2, 0.2)   | 0         | 0               |
      # (0.2, 0.12)  | 0         | 1               |
      # (4.0, 0.13)  | 1         | -1              |
      # (4.0, 0.13)  | 1         | 1               |
      partition_ids = array_ops.constant([0, 0, 0, 1, 1], dtype=dtypes.int32)
      # We have only 1 dimension in our sparse feature column.
      bucket_ids = array_ops.constant([-1, 0, 1, -1, 1], dtype=dtypes.int64)
      dimension_ids = array_ops.constant([0, 0, 0, 0, 0], dtype=dtypes.int64)
      bucket_ids = array_ops.stack([bucket_ids, dimension_ids], axis=1)

      gradients = array_ops.constant([1.8, 2.4, 0.4, 8.0, 8.0])
      hessians = array_ops.constant([0.78, 0.4, 0.24, 0.26, 0.26])
      bucket_boundaries = array_ops.constant([0.3, 0.52])
      partitions, gains, splits = (
          split_handler_ops.build_sparse_inequality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              bucket_ids=bucket_ids,
              gradients=gradients,
              hessians=hessians,
              bucket_boundaries=bucket_boundaries,
              l1_regularization=0,
              l2_regularization=2,
              tree_complexity_regularization=0,
              min_node_weight=0,
              feature_column_group_id=0,
              bias_feature_id=-1,
              class_id=-1,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = (sess.run([partitions, gains, splits]))
    self.assertAllEqual([0, 1], partitions)
    self.assertEqual(2, len(splits))
    # Check the split on partition 0.
    # -(0.2 + 1.2) / (0.12 + 0.2 + 2)
    expected_left_weight = -0.603448275862069
    # (0.2 + 1.2) ** 2 / (0.12 + 0.2 + 2)
    expected_left_gain = 0.8448275862068965
    # 0.5 / (0.07 + 2)
    expected_right_weight = 0.24154589371980678
    # 0.5 ** 2 / (0.07 + 2)
    expected_right_gain = 0.12077294685990339
    # (0.2 + 1.2 - 0.5) ** 2 /  (0.12 + 0.2 + 0.07 + 2)
    expected_bias_gain = 0.3389121338912133

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_right
    self.assertAllClose(
        expected_left_gain + expected_right_gain - expected_bias_gain, gains[0])

    self.assertAllClose([expected_left_weight], left_child.value)

    self.assertAllClose([expected_right_weight], right_child.value)

    self.assertEqual(0, split_node.split.feature_column)
    # Sparse is one dimensional.
    self.assertEqual(0, split_node.split.dimension_id)

    self.assertAllClose(0.52, split_node.split.threshold)

    # Check the split on partition 1.
    expected_left_weight = -1.8779342723004695
    expected_right_weight = 0

    # Verify candidate for partition 1, there's only one active bucket here
    # so zero gain is expected.
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_left

    self.assertAllClose(0.0, gains[1])

    self.assertAllClose([expected_left_weight], left_child.value)

    self.assertAllClose([expected_right_weight], right_child.value)

    self.assertEqual(0, split_node.split.feature_column)
    # Sparse is one dimensional.
    self.assertEqual(0, split_node.split.dimension_id)

    self.assertAllClose(0.52, split_node.split.threshold)

  def testMakeSparseSplitAllEmptyDimensions(self):
    """Tests split handler op when all dimensions have only bias bucket id."""
    with self.test_session() as sess:
      # The data looks like the following after dividing by number of steps (2).
      # Gradients    | Partition | Dimension | bucket ID       |
      # (0.9, 0.39)  | 0         |    0      |  -1             |
      # (4.0, 0.13)  | 1         |    0      |  -1             |
      partition_ids = array_ops.constant([0, 1], dtype=dtypes.int32)
      # We have only 1 dimension in our sparse feature column.
      bucket_ids = array_ops.constant([[-1, 0], [-1, 0]], dtype=dtypes.int64)
      gradients = array_ops.constant([1.8, 8.0])
      hessians = array_ops.constant([0.78, 0.26])
      bucket_boundaries = array_ops.constant([0.3, 0.52])
      partitions, gains, splits = (
          split_handler_ops.build_sparse_inequality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              bucket_ids=bucket_ids,
              gradients=gradients,
              hessians=hessians,
              bucket_boundaries=bucket_boundaries,
              l1_regularization=0,
              l2_regularization=2,
              tree_complexity_regularization=0,
              min_node_weight=0,
              feature_column_group_id=0,
              bias_feature_id=-1,
              class_id=-1,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = (sess.run([partitions, gains, splits]))
    self.assertEqual(0, len(partitions))
    self.assertEqual(0, len(splits))

  def testMakeSparseMultidimensionalSplit(self):
    """Tests split handler op."""
    with self.test_session() as sess:
      # Num of steps is 2.
      # The feature column is three dimensional.
      # First dimension has bias bucket only, the second has bias bucket and
      # two valid buckets, the third has just one bias bucket and one valid
      # bucket.
      # Gradients    | Partition | Dimension | bucket ID       |
      # (0.9, 0.39)  |    0      |     0     |     -1          |
      # (1.2, 0.2)   |    0      |     1     |      0          |
      # (0.2, 0.12)  |    0      |     1     |      2          |
      # (0.1, 0.1)   |    0      |     2     |      3          |
      # Now second node - nothing interesting there, just one dimension.
      # Second node has the same bucket ids for all dimensions.
      # (4.0, 0.13)  |    1      |     0     |     -1          |
      # (4.0, 0.13)  |    1      |     2     |      3          |

      # Tree node ids.
      partition_ids = array_ops.constant([0, 0, 0, 0, 1, 1], dtype=dtypes.int32)

      dimension_ids = array_ops.constant([0, 1, 1, 2, 0, 2], dtype=dtypes.int64)
      bucket_ids = array_ops.constant([-1, 0, 2, 3, -1, 3], dtype=dtypes.int64)
      bucket_ids = array_ops.stack([bucket_ids, dimension_ids], axis=1)

      gradients = array_ops.constant([1.8, 2.4, 0.4, 0.2, 8.0, 8.0])
      hessians = array_ops.constant([0.78, 0.4, 0.24, 0.2, 0.26, 0.26])
      bucket_boundaries = array_ops.constant([0.3, 0.52, 0.58, 0.6])
      partitions, gains, splits = (
          split_handler_ops.build_sparse_inequality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              bucket_ids=bucket_ids,
              gradients=gradients,
              hessians=hessians,
              bucket_boundaries=bucket_boundaries,
              l1_regularization=0,
              l2_regularization=2,
              tree_complexity_regularization=0,
              min_node_weight=0,
              feature_column_group_id=0,
              bias_feature_id=-1,
              class_id=-1,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = (sess.run([partitions, gains, splits]))
    self.assertAllEqual([0, 1], partitions)
    self.assertEqual(2, len(splits))
    # Check the split on node 0 - it should split on second dimension
    # -(0.2 + 1.2) / (0.12 + 0.2 + 2)
    expected_left_weight = -0.603448275862069
    # (0.2 + 1.2) ** 2 / (0.12 + 0.2 + 2)
    expected_left_gain = 0.8448275862068965
    # 0.5 / (0.07 + 2)
    expected_right_weight = 0.24154589371980678
    # 0.5 ** 2 / (0.07 + 2)
    expected_right_gain = 0.12077294685990339
    # (0.2 + 1.2 - 0.5) ** 2 /  (0.12 + 0.2 + 0.07 + 2)
    expected_bias_gain = 0.3389121338912133

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_right
    self.assertAllClose(
        expected_left_gain + expected_right_gain - expected_bias_gain, gains[0])

    self.assertAllClose([expected_left_weight], left_child.value)

    self.assertAllClose([expected_right_weight], right_child.value)

    self.assertEqual(0, split_node.split.feature_column)
    # Split happened on second dimension.
    self.assertEqual(1, split_node.split.dimension_id)

    self.assertAllClose(0.58, split_node.split.threshold)

    # Check the split on partition 1.
    expected_left_weight = -1.8779342723004695
    expected_right_weight = 0

    # Verify candidate for partition 1, there's only one active bucket here
    # so zero gain is expected.
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_left

    self.assertAllClose(0.0, gains[1])

    self.assertAllClose([expected_left_weight], left_child.value)

    self.assertAllClose([expected_right_weight], right_child.value)

    self.assertEqual(0, split_node.split.feature_column)
    self.assertEqual(2, split_node.split.dimension_id)

    self.assertAllClose(0.6, split_node.split.threshold)

  def testMakeMulticlassSparseSplit(self):
    """Tests split handler op."""
    with self.test_session() as sess:
      partition_ids = array_ops.constant([0, 0, 0, 1, 1], dtype=dtypes.int32)
    bucket_ids = array_ops.constant(
        [[-1, 0], [0, 0], [1, 0], [-1, 0], [1, 0]], dtype=dtypes.int64)
    gradients = array_ops.constant([[1.8, 3.5], [2.4, 1.0], [0.4, 4.0],
                                    [8.0, 3.1], [8.0, 0.8]])

    hessian_0 = [[0.78, 1], [12, 1]]
    hessian_1 = [[0.4, 1], [1, 1]]
    hessian_2 = [[0.24, 1], [1, 1]]
    hessian_3 = [[0.26, 1], [1, 1]]
    hessian_4 = [[0.26, 1], [1, 1]]

    hessians = array_ops.constant(
        [hessian_0, hessian_1, hessian_2, hessian_3, hessian_4])
    bucket_boundaries = array_ops.constant([0.3, 0.52])
    partitions, gains, splits = (
        split_handler_ops.build_sparse_inequality_splits(
            num_minibatches=2,
            partition_ids=partition_ids,
            bucket_ids=bucket_ids,
            gradients=gradients,
            hessians=hessians,
            bucket_boundaries=bucket_boundaries,
            l1_regularization=0,
            l2_regularization=2,
            tree_complexity_regularization=0,
            min_node_weight=0,
            feature_column_group_id=0,
            bias_feature_id=-1,
            class_id=-1,
            multiclass_strategy=learner_pb2.LearnerConfig.FULL_HESSIAN))
    partitions, gains, splits = (sess.run([partitions, gains, splits]))

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_right

    # Each leaf has 2 element vector.
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(2, len(right_child.value))

    self.assertEqual(0, split_node.split.feature_column)
    self.assertAllClose(0.52, split_node.split.threshold)

  def testMakeCategoricalEqualitySplit(self):
    """Tests split handler op for categorical equality split."""
    with self.test_session() as sess:
      # The data looks like the following after dividing by number of steps (2).
      # Gradients    | Partition | Feature ID     |
      # (0.9, 0.39)  | 0         | -1             |
      # (0.2, 0.12)  | 0         | 1              |
      # (1.4, 0.32)  | 0         | 2              |
      # (4.0, 0.13)  | 1         | -1             |
      # (4.0, 0.13)  | 1         | 1              |
      gradients = [1.8, 0.4, 2.8, 8.0, 8.0]
      hessians = [0.78, 0.24, 0.64, 0.26, 0.26]
      partition_ids = [0, 0, 0, 1, 1]
      feature_ids = array_ops.constant(
          [[-1, 0], [1, 0], [2, 0], [-1, 0], [1, 0]], dtype=dtypes.int64)
      partitions, gains, splits = (
          split_handler_ops.build_categorical_equality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              feature_ids=feature_ids,
              gradients=gradients,
              hessians=hessians,
              l1_regularization=0.1,
              l2_regularization=1,
              tree_complexity_regularization=0,
              min_node_weight=0,
              feature_column_group_id=0,
              bias_feature_id=-1,
              class_id=-1,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = sess.run([partitions, gains, splits])
    self.assertAllEqual([0, 1], partitions)

    # Check the split on partition 0.
    # -(0.2 + 1.2 - 0.1) / (0.12 + 0.2 + 1)
    expected_left_weight = -0.9848484848484846

    # (0.2 + 1.2 - 0.1) ** 2 / (0.12 + 0.2 + 1)
    expected_left_gain = 1.2803030303030298

    # -(-0.5 + 0.1) / (0.07 + 1)
    expected_right_weight = 0.37383177570093457

    # (-0.5 + 0.1) ** 2 / (0.07 + 1)
    expected_right_gain = 0.14953271028037385

    # (0.2 + -0.5 + 1.2 - 0.1) ** 2 / (0.12 + 0.07 + 0.2 + 1)
    expected_bias_gain = 0.46043165467625885

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.categorical_id_binary_split

    self.assertEqual(0, split_node.feature_column)

    self.assertEqual(2, split_node.feature_id)

    self.assertAllClose(
        expected_left_gain + expected_right_gain - expected_bias_gain, gains[0],
        0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    # Check the split on partition 1.
    # (-4 + 0.1) / (0.13 + 1)
    expected_left_weight = -3.4513274336283186
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_left_gain = 13.460176991150442
    expected_right_weight = 0
    expected_right_gain = 0
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_bias_gain = 13.460176991150442

    # Verify candidate for partition 1, there's only one active feature here
    # so zero gain is expected.
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.categorical_id_binary_split
    self.assertAllClose(0.0, gains[1], 0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    self.assertEqual(0, split_node.feature_column)

    self.assertEqual(1, split_node.feature_id)

  def testMakeMulticlassCategoricalEqualitySplit(self):
    """Tests split handler op for categorical equality split in multiclass."""
    with self.test_session() as sess:
      gradients = array_ops.constant([[1.8, 3.5], [2.4, 1.0], [0.4, 4.0],
                                      [9.0, 3.1], [3.0, 0.8]])

      hessian_0 = [[0.78, 1], [12, 1]]
      hessian_1 = [[0.4, 1], [1, 1]]
      hessian_2 = [[0.24, 1], [1, 1]]
      hessian_3 = [[0.16, 2], [-1, 1]]
      hessian_4 = [[0.6, 1], [2, 1]]

      hessians = array_ops.constant(
          [hessian_0, hessian_1, hessian_2, hessian_3, hessian_4])
      partition_ids = [0, 0, 0, 1, 1]
      feature_ids = array_ops.constant(
          [[-1, 0], [1, 0], [2, 0], [-1, 0], [1, 0]], dtype=dtypes.int64)
      partitions, gains, splits = (
          split_handler_ops.build_categorical_equality_splits(
              num_minibatches=2,
              partition_ids=partition_ids,
              feature_ids=feature_ids,
              gradients=gradients,
              hessians=hessians,
              l1_regularization=0.1,
              l2_regularization=1,
              tree_complexity_regularization=0,
              min_node_weight=0,
              feature_column_group_id=0,
              bias_feature_id=-1,
              class_id=-1,
              multiclass_strategy=learner_pb2.LearnerConfig.FULL_HESSIAN))
      partitions, gains, splits = sess.run([partitions, gains, splits])
    self.assertAllEqual([0, 1], partitions)

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.categorical_id_binary_split

    # Each leaf has 2 element vector.
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(2, len(right_child.value))

    self.assertEqual(0, split_node.feature_column)
    self.assertEqual(1, split_node.feature_id)

  def testMakeCategoricalEqualitySplitEmptyInput(self):
    with self.test_session() as sess:
      gradients = []
      hessians = []
      partition_ids = []
      feature_ids = [[]]
      partitions, gains, splits = (
          split_handler_ops.build_categorical_equality_splits(
              num_minibatches=0,
              partition_ids=partition_ids,
              feature_ids=feature_ids,
              gradients=gradients,
              hessians=hessians,
              l1_regularization=0.1,
              l2_regularization=1,
              tree_complexity_regularization=0,
              min_node_weight=0,
              feature_column_group_id=0,
              bias_feature_id=-1,
              class_id=-1,
              multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS))
      partitions, gains, splits = (sess.run([partitions, gains, splits]))
    self.assertEqual(0, len(partitions))
    self.assertEqual(0, len(gains))
    self.assertEqual(0, len(splits))


if __name__ == "__main__":
  googletest.main()
