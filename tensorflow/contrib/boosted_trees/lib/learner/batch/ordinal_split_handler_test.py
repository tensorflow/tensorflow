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
"""Test for checking stats accumulator related ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.boosted_trees.lib.learner.batch import ordinal_split_handler
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.proto import split_info_pb2

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


def get_empty_tensors(gradient_shape, hessian_shape):
  empty_hess_shape = [1] + hessian_shape.as_list()
  empty_grad_shape = [1] + gradient_shape.as_list()

  empty_gradients = constant_op.constant(
      [], dtype=dtypes.float32, shape=empty_grad_shape)
  empty_hessians = constant_op.constant(
      [], dtype=dtypes.float32, shape=empty_hess_shape)

  return empty_gradients, empty_hessians


class DenseSplitHandlerTest(test_util.TensorFlowTestCase):

  def testGenerateFeatureSplitCandidates(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Dense Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1              |
      # i1      |  (-0.5, 0.07) | 0         | 1              |
      # i2      |  (1.2, 0.2)   | 0         | 0              |
      # i3      |  (4.0, 0.13)  | 1         | 1              |
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      class_id = -1

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.1,
          l2_regularization=1.,
          tree_complexity_regularization=0.,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=10,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([0, 1], partitions)

    # Check the split on partition 0.
    # -(1.2 - 0.1) / (0.2 + 1)
    expected_left_weight = -0.91666

    # expected_left_weight * -(1.2 - 0.1)
    expected_left_gain = 1.0083333333333331

    # (-0.5 + 0.2 + 0.1) / (0.19 + 1)
    expected_right_weight = 0.1680672

    # expected_right_weight * -(-0.5 + 0.2 + 0.1))
    expected_right_gain = 0.033613445378151252

    # (0.2 + -0.5 + 1.2 - 0.1) ** 2 / (0.12 + 0.07 + 0.2 + 1)
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
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_left_gain = 13.460176991150442
    expected_right_weight = 0
    expected_right_gain = 0
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_bias_gain = 13.460176991150442

    # Verify candidate for partition 1, there's only one active bucket here
    # so zero gain is expected.
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    self.assertAllClose(0.0, gains[1], 0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    self.assertEqual(0, split_node.feature_column)

    self.assertAllClose(0.52, split_node.threshold, 0.00001)

  def testObliviousFeatureSplitGeneration(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Dense Quantile |
      # i0      |  (0.2, 0.12)  | 1         | 3              |
      # i1      |  (-0.5, 0.07) | 1         | 3              |
      # i2      |  (1.2, 0.2)   | 1         | 1              |
      # i3      |  (4.0, 0.13)  | 2         | 2              |
      dense_column = array_ops.placeholder(
          dtypes.float32, shape=(4, 1), name="dense_column")
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([1, 1, 1, 2], dtype=dtypes.int32)
      class_id = -1

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.1,
          l2_regularization=1.,
          tree_complexity_regularization=0.,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=10,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          weak_learner_type=learner_pb2.LearnerConfig.OBLIVIOUS_DECISION_TREE)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
        # Forcing the creation of four buckets.
        are_splits_ready = sess.run(
            [are_splits_ready],
            feed_dict={dense_column: [[0.2], [0.62], [0.3], [0.52]]})[0]

      update_2 = split_handler.update_stats_sync(
          1,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        # Only using the last three buckets.
        are_splits_ready2, partitions, gains, splits = (
            sess.run(
                [are_splits_ready2, partitions, gains, splits],
                feed_dict={dense_column: [[0.62], [0.62], [0.3], [0.52]]}))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([1, 2], partitions)

    oblivious_split_info = split_info_pb2.ObliviousSplitInfo()
    oblivious_split_info.ParseFromString(splits[0])
    split_node = oblivious_split_info.split_node
    split_node = split_node.oblivious_dense_float_binary_split
    self.assertAllClose(0.3, split_node.threshold, 0.00001)
    self.assertEqual(0, split_node.feature_column)

    # Check the split on partition 1.
    # -(1.2 - 0.1) / (0.2 + 1)
    expected_left_weight_1 = -0.9166666666666666

    # expected_left_weight_1 * -(1.2 - 0.1)
    expected_left_gain_1 = 1.008333333333333

    # (-0.5 + 0.2 + 0.1) / (0.19 + 1)
    expected_right_weight_1 = 0.1680672

    # expected_right_weight_1 * -(-0.5 + 0.2 + 0.1))
    expected_right_gain_1 = 0.033613445378151252

    # (0.2 + -0.5 + 1.2 - 0.1) ** 2 / (0.12 + 0.07 + 0.2 + 1)
    expected_bias_gain_1 = 0.46043165467625896

    left_child = oblivious_split_info.children[0].vector
    right_child = oblivious_split_info.children[1].vector

    self.assertAllClose([expected_left_weight_1], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight_1], right_child.value, 0.00001)

    # Check the split on partition 2.
    expected_left_weight_2 = 0
    expected_left_gain_2 = 0
    # -(4 - 0.1) / (0.13 + 1)
    expected_right_weight_2 = -3.4513274336283186
    # expected_right_weight_2 * -(4 - 0.1)
    expected_right_gain_2 = 13.460176991150442
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_bias_gain_2 = 13.460176991150442

    left_child = oblivious_split_info.children[2].vector
    right_child = oblivious_split_info.children[3].vector

    self.assertAllClose([expected_left_weight_2], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight_2], right_child.value, 0.00001)

    # The layer gain is the sum of the gains of each partition
    layer_gain = (
        expected_left_gain_1 + expected_right_gain_1 - expected_bias_gain_1) + (
            expected_left_gain_2 + expected_right_gain_2 - expected_bias_gain_2)
    self.assertAllClose(layer_gain, gains[0], 0.00001)

    # We have examples in both partitions, then we get both ids.
    self.assertEqual(2, len(oblivious_split_info.children_parent_id))
    self.assertEqual(1, oblivious_split_info.children_parent_id[0])
    self.assertEqual(2, oblivious_split_info.children_parent_id[1])

  def testGenerateFeatureSplitCandidatesLossUsesSumReduction(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Dense Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1              |
      # i1      |  (-0.5, 0.07) | 0         | 1              |
      # i2      |  (1.2, 0.2)   | 0         | 0              |
      # i3      |  (4.0, 0.13)  | 1         | 1              |
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      class_id = -1

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.2,
          l2_regularization=2.,
          tree_complexity_regularization=0.,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=10,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          loss_uses_sum_reduction=True)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
        update_3 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2, update_3]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([0, 1], partitions)

    # Check the split on partition 0.
    # -(2.4 - 0.2) / (0.4 + 2)
    expected_left_weight = -0.91666

    # expected_left_weight * -(2.4 - 0.2)
    expected_left_gain = 2.016666666666666

    # -(-1 + 0.4 + 0.2) / (0.38 + 2)
    expected_right_weight = 0.1680672

    # expected_right_weight * -(-1 + 0.4 + 0.2)
    expected_right_gain = 0.0672268907563025

    # (0.2 + -0.5 + 1.2 - 0.1) ** 2 / (0.12 + 0.07 + 0.2 + 1)
    expected_bias_gain = 0.9208633093525178

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
    # (-8 + 0.2) / (0.26 + 2)
    expected_left_weight = -3.4513274336283186
    expected_right_weight = 0

    # Verify candidate for partition 1, there's only one active bucket here
    # so zero gain is expected.
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    self.assertAllClose(0.0, gains[1], 0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    self.assertEqual(0, split_node.feature_column)

    self.assertAllClose(0.52, split_node.threshold, 0.00001)

  def testGenerateFeatureSplitCandidatesMulticlassFullHessian(self):
    with self.test_session() as sess:
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      # Batch size is 4, 2 gradients per each instance.
      gradients = array_ops.constant(
          [[0.2, 0.1], [-0.5, 0.2], [1.2, 3.4], [4.0, -3.5]], shape=[4, 2])
      # 2x2 matrix for each instance
      hessian_0 = [[0.12, 0.02], [0.3, 0.11]]
      hessian_1 = [[0.07, -0.2], [-0.5, 0.2]]
      hessian_2 = [[0.2, -0.23], [-0.8, 0.9]]
      hessian_3 = [[0.13, -0.3], [-1.5, 2.2]]

      hessians = array_ops.constant(
          [hessian_0, hessian_1, hessian_2, hessian_3])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      class_id = -1

      gradient_shape = tensor_shape.TensorShape([2])
      hessian_shape = tensor_shape.TensorShape([2, 2])

      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.,
          l2_regularization=1.,
          tree_complexity_regularization=0.,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=3,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.FULL_HESSIAN)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

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

  def testGenerateFeatureSplitCandidatesMulticlassDiagonalHessian(self):
    with self.test_session() as sess:
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      # Batch size is 4, 2 gradients per each instance.
      gradients = array_ops.constant(
          [[0.2, 0.1], [-0.5, 0.2], [1.2, 3.4], [4.0, -3.5]], shape=[4, 2])
      # Each hessian is a diagonal of a full hessian matrix.
      hessian_0 = [0.12, 0.11]
      hessian_1 = [0.07, 0.2]
      hessian_2 = [0.2, 0.9]
      hessian_3 = [0.13, 2.2]

      hessians = array_ops.constant(
          [hessian_0, hessian_1, hessian_2, hessian_3])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      class_id = -1

      gradient_shape = tensor_shape.TensorShape([2])
      hessian_shape = tensor_shape.TensorShape([2])

      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.,
          l2_regularization=1.,
          tree_complexity_regularization=0.,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=3,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

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

  def testGenerateFeatureSplitCandidatesInactive(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Dense Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1              |
      # i1      |  (-0.5, 0.07) | 0         | 1              |
      # i2      |  (1.2, 0.2)   | 0         | 0              |
      # i3      |  (4.0, 0.13)  | 1         | 1              |
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.1,
          l2_regularization=1.,
          tree_complexity_regularization=0.,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=10,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, False]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([False, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)
    # The handler was inactive, so it shouldn't return any splits.
    self.assertEqual(len(partitions), 0)
    self.assertEqual(len(gains), 0)
    self.assertEqual(len(splits), 0)

  def testGenerateFeatureSplitCandidatesWithTreeComplexity(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Dense Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1              |
      # i1      |  (-0.5, 0.07) | 0         | 1              |
      # i2      |  (1.2, 0.2)   | 0         | 0              |
      # i3      |  (4.0, 0.13)  | 1         | 1              |
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.1,
          l2_regularization=1.,
          tree_complexity_regularization=0.5,
          min_node_weight=0.,
          epsilon=0.001,
          num_quantiles=10,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([0, 1], partitions)

    # Check the split on partition 0.
    # -(1.2 - 0.1) / (0.2 + 1)
    expected_left_weight = -0.91666

    # expected_left_weight * -(1.2 - 0.1)
    expected_left_gain = 1.0083333333333331

    # (-0.5 + 0.2 + 0.1) / (0.19 + 1)
    expected_right_weight = 0.1680672

    # expected_right_weight * -(-0.5 + 0.2 + 0.1))
    expected_right_gain = 0.033613445378151252

    # (0.2 + -0.5 + 1.2 - 0.1) ** 2 / (0.12 + 0.07 + 0.2 + 1)
    expected_bias_gain = 0.46043165467625885

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    # Make sure the gain is subtracted by the tree complexity regularization.
    self.assertAllClose(
        expected_left_gain + expected_right_gain - expected_bias_gain - 0.5,
        gains[0], 0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    self.assertEqual(0, split_node.feature_column)

    self.assertAllClose(0.3, split_node.threshold, 0.00001)

    # Check the split on partition 1.
    # (-4 + 0.1) / (0.13 + 1)
    expected_left_weight = -3.4513274336283186
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_left_gain = 13.460176991150442
    expected_right_weight = 0
    expected_right_gain = 0
    # (-4 + 0.1) ** 2 / (0.13 + 1)
    expected_bias_gain = 13.460176991150442

    # Verify candidate for partition 1, there's only one active bucket here
    # so -0.5 gain is expected (because of tree complexity.
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    self.assertAllClose(-0.5, gains[1], 0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    self.assertEqual(0, split_node.feature_column)

    self.assertAllClose(0.52, split_node.threshold, 0.00001)

  def testGenerateFeatureSplitCandidatesWithMinNodeWeight(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Dense Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1              |
      # i1      |  (-0.5, 0.07) | 0         | 1              |
      # i2      |  (1.2, 0.2)   | 0         | 0              |
      # i3      |  (4.0, 2.0)   | 1         | 1              |
      dense_column = array_ops.constant([0.52, 0.52, 0.3, 0.52])
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 2])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.DenseSplitHandler(
          l1_regularization=0.1,
          l2_regularization=1.,
          tree_complexity_regularization=0.5,
          min_node_weight=1.5,
          epsilon=0.001,
          num_quantiles=10,
          feature_column_group_id=0,
          dense_float_column=dense_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([0, 1], partitions)

    # Check the gain on partition 0 to be -0.5.
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    # Make sure the gain is subtracted by the tree complexity regularization.
    self.assertAllClose(-0.5, gains[0], 0.00001)

    self.assertEqual(0, split_node.feature_column)

    # Check the split on partition 1.
    # (-4 + 0.1) / (2 + 1)
    expected_left_weight = -1.3
    expected_right_weight = 0

    # Verify candidate for partition 1, there's only one active bucket here
    # so -0.5 gain is expected (because of tree complexity.
    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.dense_float_binary_split
    self.assertAllClose(-0.5, gains[1], 0.00001)

    self.assertAllClose([expected_left_weight], left_child.value, 0.00001)

    self.assertAllClose([expected_right_weight], right_child.value, 0.00001)

    self.assertEqual(0, split_node.feature_column)

    self.assertAllClose(0.52, split_node.threshold, 0.00001)


class SparseSplitHandlerTest(test_util.TensorFlowTestCase):

  def testGenerateFeatureSplitCandidates(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Sparse Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1               |
      # i1      |  (-0.5, 0.07) | 0         | N/A             |
      # i2      |  (1.2, 0.2)   | 0         | 0               |
      # i3      |  (4.0, 0.13)  | 1         | 1               |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      example_partitions = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      indices = array_ops.constant([[0, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
      values = array_ops.constant([0.52, 0.3, 0.52])
      sparse_column = sparse_tensor.SparseTensor(indices, values, [4, 1])

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          example_partitions,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([0, 1], partitions)
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

    self.assertAllClose(0.52, split_node.split.threshold)

  def testGenerateFeatureSplitCandidatesLossUsesSumReduction(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Sparse Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1               |
      # i1      |  (-0.5, 0.07) | 0         | N/A             |
      # i2      |  (1.2, 0.2)   | 0         | 0               |
      # i3      |  (4.0, 0.13)  | 1         | 1               |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      example_partitions = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      indices = array_ops.constant([[0, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
      values = array_ops.constant([0.52, 0.3, 0.52])
      sparse_column = sparse_tensor.SparseTensor(indices, values, [4, 1])

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=4.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          loss_uses_sum_reduction=True)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          example_partitions,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]
      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
        update_3 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2, update_3]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([0, 1], partitions)
    # Check the split on partition 0.
    # -(0.4 + 2.4) / (0.24 + 0.4 + 4)
    expected_left_weight = -0.603448275862069
    # (0.4 + 2.4) ** 2 / (0.24 + 0.4 + 4)
    expected_left_gain = 1.689655172413793
    # 1 / (0.14 + 4)
    expected_right_weight = 0.24154589371980678
    # 1 ** 2 / (0.14 + 4)
    expected_right_gain = 0.24154589371980678
    # (0.4 + 2.4 - 1) ** 2 /  (0.24 + 0.4 + 0.14 + 4)
    expected_bias_gain = 0.6778242677824265

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

    self.assertAllClose(0.52, split_node.split.threshold)

  def testGenerateFeatureSplitCandidatesMulticlassFullHessian(self):
    with self.test_session() as sess:
      # Batch is 4, 2 classes
      gradients = array_ops.constant([[0.2, 1.4], [-0.5, 0.1], [1.2, 3],
                                      [4.0, -3]])
      # 2x2 matrix for each instance
      hessian_0 = [[0.12, 0.02], [0.3, 0.11]]
      hessian_1 = [[0.07, -0.2], [-0.5, 0.2]]
      hessian_2 = [[0.2, -0.23], [-0.8, 0.9]]
      hessian_3 = [[0.13, -0.3], [-1.5, 2.2]]
      hessians = array_ops.constant(
          [hessian_0, hessian_1, hessian_2, hessian_3])

      example_partitions = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      indices = array_ops.constant([[0, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
      values = array_ops.constant([0.52, 0.3, 0.52])
      sparse_column = sparse_tensor.SparseTensor(indices, values, [4, 1])

      gradient_shape = tensor_shape.TensorShape([2])
      hessian_shape = tensor_shape.TensorShape([2, 2])
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.FULL_HESSIAN)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          example_partitions,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

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

    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_left
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(0, split_node.split.feature_column)
    self.assertAllClose(0.52, split_node.split.threshold)

  def testGenerateFeatureSplitCandidatesMulticlassDiagonalHessian(self):
    with self.test_session() as sess:
      # Batch is 4, 2 classes
      gradients = array_ops.constant([[0.2, 1.4], [-0.5, 0.1], [1.2, 3],
                                      [4.0, -3]])
      # Each hessian is a diagonal from a full hessian matrix.
      hessian_0 = [0.12, 0.11]
      hessian_1 = [0.07, 0.2]
      hessian_2 = [0.2, 0.9]
      hessian_3 = [0.13, 2.2]
      hessians = array_ops.constant(
          [hessian_0, hessian_1, hessian_2, hessian_3])

      example_partitions = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      indices = array_ops.constant([[0, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
      values = array_ops.constant([0.52, 0.3, 0.52])
      sparse_column = sparse_tensor.SparseTensor(indices, values, [4, 1])

      gradient_shape = tensor_shape.TensorShape([2])
      hessian_shape = tensor_shape.TensorShape([2])
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          example_partitions,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

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

    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.sparse_float_binary_split_default_left
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(0, split_node.split.feature_column)
    self.assertAllClose(0.52, split_node.split.threshold)

  def testGenerateFeatureSplitCandidatesInactive(self):
    with self.test_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Sparse Quantile |
      # i0      |  (0.2, 0.12)  | 0         | 1               |
      # i1      |  (-0.5, 0.07) | 0         | N/A             |
      # i2      |  (1.2, 0.2)   | 0         | 0               |
      # i3      |  (4.0, 0.13)  | 1         | 1               |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      example_partitions = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)
      indices = array_ops.constant([[0, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
      values = array_ops.constant([0.52, 0.3, 0.52])
      sparse_column = sparse_tensor.SparseTensor(indices, values, [4, 1])

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          example_partitions,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, False]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([False, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)
    # The handler was inactive so it shouldn't any splits.
    self.assertEqual(len(partitions), 0)
    self.assertEqual(len(gains), 0)
    self.assertEqual(len(splits), 0)

  def testEmpty(self):
    with self.test_session() as sess:
      indices = array_ops.constant([], dtype=dtypes.int64, shape=[0, 2])
      # No values in this feature column in this mini-batch.
      values = array_ops.constant([], dtype=dtypes.float32)
      sparse_column = sparse_tensor.SparseTensor(indices, values, [4, 1])

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            partition_ids,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)
    self.assertEqual(len(partitions), 0)
    self.assertEqual(len(gains), 0)
    self.assertEqual(len(splits), 0)

  def testEmptyBuckets(self):
    """Test that reproduces the case when quantile buckets were empty."""
    with self.test_session() as sess:
      sparse_column = array_ops.sparse_placeholder(dtypes.float32)

      # We have two batches - at first, a sparse feature is empty.
      empty_indices = array_ops.constant([], dtype=dtypes.int64, shape=[0, 2])
      empty_values = array_ops.constant([], dtype=dtypes.float32)
      empty_sparse_column = sparse_tensor.SparseTensor(empty_indices,
                                                       empty_values, [4, 2])
      empty_sparse_column = empty_sparse_column.eval(session=sess)

      # For the second batch, the sparse feature is not empty.
      non_empty_indices = array_ops.constant(
          [[0, 0], [2, 1], [3, 2]], dtype=dtypes.int64, shape=[3, 2])
      non_empty_values = array_ops.constant(
          [0.52, 0.3, 0.52], dtype=dtypes.float32)
      non_empty_sparse_column = sparse_tensor.SparseTensor(
          non_empty_indices, non_empty_values, [4, 2])
      non_empty_sparse_column = non_empty_sparse_column.eval(session=sess)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([4, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

        # First, calculate quantiles and try to update on an empty data for a
        # feature.
        are_splits_ready = (
            sess.run(
                are_splits_ready,
                feed_dict={sparse_column: empty_sparse_column}))
        self.assertFalse(are_splits_ready)

      update_2 = split_handler.update_stats_sync(
          1,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))

        # Now the feature in the second batch is not empty, but buckets
        # calculated on the first batch are empty.
        are_splits_ready2, partitions, gains, splits = (
            sess.run(
                [are_splits_ready2, partitions, gains, splits],
                feed_dict={sparse_column: non_empty_sparse_column}))
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)
    # Since the buckets were empty, we can't calculate the splits.
    self.assertEqual(len(partitions), 0)
    self.assertEqual(len(gains), 0)
    self.assertEqual(len(splits), 0)

  def testDegenerativeCase(self):
    with self.test_session() as sess:
      # One data example only, one leaf and thus one quantile bucket.The same
      # situation is when all examples have the same values. This case was
      # causing before a failure.
      gradients = array_ops.constant([0.2])
      hessians = array_ops.constant([0.12])
      example_partitions = array_ops.constant([1], dtype=dtypes.int32)
      indices = array_ops.constant([[0, 0]], dtype=dtypes.int64)
      values = array_ops.constant([0.58])
      sparse_column = sparse_tensor.SparseTensor(indices, values, [1, 1])

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = ordinal_split_handler.SparseSplitHandler(
          l1_regularization=0.0,
          l2_regularization=2.0,
          tree_complexity_regularization=0.0,
          min_node_weight=0.0,
          epsilon=0.01,
          num_quantiles=2,
          feature_column_group_id=0,
          sparse_float_column=sparse_column,
          init_stamp_token=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS)
      resources.initialize_resources(resources.shared_resources()).run()

      empty_gradients, empty_hessians = get_empty_tensors(
          gradient_shape, hessian_shape)
      example_weights = array_ops.ones([1, 1], dtypes.float32)

      update_1 = split_handler.update_stats_sync(
          0,
          example_partitions,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1]):
        are_splits_ready = split_handler.make_splits(
            np.int64(0), np.int64(1), class_id)[0]

      with ops.control_dependencies([are_splits_ready]):
        update_2 = split_handler.update_stats_sync(
            1,
            example_partitions,
            gradients,
            hessians,
            empty_gradients,
            empty_hessians,
            example_weights,
            is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_2]):
        are_splits_ready2, partitions, gains, splits = (
            split_handler.make_splits(np.int64(1), np.int64(2), class_id))
        are_splits_ready, are_splits_ready2, partitions, gains, splits = (
            sess.run([
                are_splits_ready, are_splits_ready2, partitions, gains, splits
            ]))

    # During the first iteration, inequality split handlers are not going to
    # have any splits. Make sure that we return not_ready in that case.
    self.assertFalse(are_splits_ready)
    self.assertTrue(are_splits_ready2)

    self.assertAllEqual([1], partitions)
    self.assertAllEqual([0.0], gains)

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])
    split_node = split_info.split_node.sparse_float_binary_split_default_left

    self.assertEqual(0, split_node.split.feature_column)

    self.assertAllClose(0.58, split_node.split.threshold)


if __name__ == "__main__":
  googletest.main()
