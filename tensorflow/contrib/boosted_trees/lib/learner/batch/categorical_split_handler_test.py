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

from tensorflow.contrib.boosted_trees.lib.learner.batch import categorical_split_handler
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


class EqualitySplitHandlerTest(test_util.TensorFlowTestCase):

  def testGenerateFeatureSplitCandidates(self):
    with self.cached_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Feature ID     |
      # i0      |  (0.2, 0.12)  | 0         | 1,2            |
      # i1      |  (-0.5, 0.07) | 0         |                |
      # i2      |  (1.2, 0.2)   | 0         | 2              |
      # i3      |  (4.0, 0.13)  | 1         | 1              |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = [0, 0, 0, 1]
      indices = [[0, 0], [0, 1], [2, 0], [3, 0]]
      values = array_ops.constant([1, 2, 2, 1], dtype=dtypes.int64)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          init_stamp_token=0)
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
      update_2 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))

      with ops.control_dependencies([update_1, update_2]):
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (sess.run(
            [are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
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

  def testObliviousFeatureSplitGeneration(self):
    with self.cached_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Feature ID     |
      # i0      |  (0.2, 0.12)  | 1         | 1              |
      # i1      |  (-0.5, 0.07) | 1         | 2              |
      # i2      |  (1.2, 0.2)   | 1         | 1              |
      # i3      |  (4.0, 0.13)  | 2         | 2              |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = [1, 1, 1, 2]
      indices = [[0, 0], [1, 0], [2, 0], [3, 0]]
      values = array_ops.constant([1, 2, 1, 2], dtype=dtypes.int64)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          init_stamp_token=0,
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
      update_2 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))

      with ops.control_dependencies([update_1, update_2]):
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (
            sess.run([are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
    self.assertAllEqual([1, 2], partitions)

    # For partition 1.
    # -(0.2 + 1.2 - 0.1) / (0.12 + 0.2 + 1)
    expected_left_weight1 = -0.9848484848484846
    # (0.2 + 1.2 - 0.1) ** 2 / (0.12 + 0.2 + 1)
    expected_left_gain1 = 1.2803030303030298

    # -(-0.5 + 0.1) / (0.07 + 1)
    expected_right_weight1 = 0.37383177570093457

    # (-0.5 + 0.1) ** 2 / (0.07 + 1)
    expected_right_gain1 = 0.14953271028037385

    # (0.2 + -0.5 + 1.2 - 0.1) ** 2 / (0.12 + 0.07 + 0.2 + 1)
    expected_bias_gain1 = 0.46043165467625885

    split_info = split_info_pb2.ObliviousSplitInfo()
    split_info.ParseFromString(splits[0])
    # Children of partition 1.
    left_child = split_info.children[0].vector
    right_child = split_info.children[1].vector
    split_node = split_info.split_node.oblivious_categorical_id_binary_split

    self.assertEqual(0, split_node.feature_column)
    self.assertEqual(1, split_node.feature_id)
    self.assertAllClose([expected_left_weight1], left_child.value, 0.00001)
    self.assertAllClose([expected_right_weight1], right_child.value, 0.00001)

    # For partition2.
    expected_left_weight2 = 0
    expected_left_gain2 = 0
    # -(4 - 0.1) / (0.13 + 1)
    expected_right_weight2 = -3.4513274336283186
    # (4 - 0.1) ** 2 / (0.13 + 1)
    expected_right_gain2 = 13.460176991150442
    # (4 - 0.1) ** 2 / (0.13 + 1)
    expected_bias_gain2 = 13.460176991150442

    # Children of partition 2.
    left_child = split_info.children[2].vector
    right_child = split_info.children[3].vector
    self.assertAllClose([expected_left_weight2], left_child.value, 0.00001)
    self.assertAllClose([expected_right_weight2], right_child.value, 0.00001)

    self.assertAllClose(
        expected_left_gain1 + expected_right_gain1 - expected_bias_gain1 +
        expected_left_gain2 + expected_right_gain2 - expected_bias_gain2,
        gains[0], 0.00001)

  def testGenerateFeatureSplitCandidatesSumReduction(self):
    with self.cached_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Feature ID     |
      # i0      |  (0.2, 0.12)  | 0         | 1,2            |
      # i1      |  (-0.5, 0.07) | 0         |                |
      # i2      |  (1.2, 0.2)   | 0         | 2              |
      # i3      |  (4.0, 0.13)  | 1         | 1              |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = [0, 0, 0, 1]
      indices = [[0, 0], [0, 1], [2, 0], [3, 0]]
      values = array_ops.constant([1, 2, 2, 1], dtype=dtypes.int64)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          init_stamp_token=0,
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
      update_2 = split_handler.update_stats_sync(
          0,
          partition_ids,
          gradients,
          hessians,
          empty_gradients,
          empty_hessians,
          example_weights,
          is_active=array_ops.constant([True, True]))
      with ops.control_dependencies([update_1, update_2]):
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (
            sess.run([are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
    self.assertAllEqual([0, 1], partitions)

    # Check the split on partition 0.
    # -(0.4 + 2.4 - 0.1) / (0.24 + 0.4 + 1)
    expected_left_weight = -1.6463414634146338

    # (0.4 + 2.4 - 0.1) ** 2 / (0.24 + 0.4 + 1)
    expected_left_gain = 4.445121951219511

    # -(-1 + 0.1) / (0.14 + 1)
    expected_right_weight = 0.789473684211

    # (-1 + 0.1) ** 2 / (0.14 + 1)
    expected_right_gain = 0.710526315789

    # (0.4 + -1 + 2.4 - 0.1) ** 2 / (0.24 + 0.14 + 0.4 + 1)
    expected_bias_gain = 1.6235955056179772

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
    # (-8 + 0.1) / (0.26 + 1)
    expected_left_weight = -6.26984126984
    # (-8 + 0.1) ** 2 / (0.26 + 1)
    expected_left_gain = 49.5317460317
    expected_right_weight = 0
    expected_right_gain = 0
    # (-8 + 0.1) ** 2 / (0.26 + 1)
    expected_bias_gain = 49.5317460317

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

  def testGenerateFeatureSplitCandidatesMulticlass(self):
    with self.cached_session() as sess:
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

      partition_ids = [0, 0, 0, 1]
      indices = [[0, 0], [0, 1], [2, 0], [3, 0]]
      values = array_ops.constant([1, 2, 2, 1], dtype=dtypes.int64)

      hessians = array_ops.constant(
          [hessian_0, hessian_1, hessian_2, hessian_3])
      partition_ids = array_ops.constant([0, 0, 0, 1], dtype=dtypes.int32)

      gradient_shape = tensor_shape.TensorShape([2])
      hessian_shape = tensor_shape.TensorShape([2, 2])
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.FULL_HESSIAN,
          init_stamp_token=0)
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
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (sess.run(
            [are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
    self.assertAllEqual([0, 1], partitions)

    split_info = split_info_pb2.SplitInfo()
    split_info.ParseFromString(splits[0])

    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.categorical_id_binary_split
    # Each leaf has 2 element vector.
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(2, len(right_child.value))
    self.assertEqual(1, split_node.feature_id)

    split_info.ParseFromString(splits[1])
    left_child = split_info.left_child.vector
    right_child = split_info.right_child.vector
    split_node = split_info.split_node.categorical_id_binary_split
    self.assertEqual(2, len(left_child.value))
    self.assertEqual(0, len(right_child.value))
    self.assertEqual(1, split_node.feature_id)

  def testEmpty(self):
    with self.cached_session() as sess:
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = [0, 0, 0, 1]
      indices = array_ops.constant([], dtype=dtypes.int64, shape=[0, 2])
      values = array_ops.constant([], dtype=dtypes.int64)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          init_stamp_token=0)
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
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (sess.run(
            [are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
    self.assertEqual(len(partitions), 0)
    self.assertEqual(len(gains), 0)
    self.assertEqual(len(splits), 0)

  def testInactive(self):
    with self.cached_session() as sess:
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = [0, 0, 0, 1]
      indices = [[0, 0], [0, 1], [2, 0], [3, 0]]
      values = array_ops.constant([1, 2, 2, 1], dtype=dtypes.int64)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          init_stamp_token=0)
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
          is_active=array_ops.constant([False, False]))
      with ops.control_dependencies([update_1]):
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (sess.run(
            [are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
    self.assertEqual(len(partitions), 0)
    self.assertEqual(len(gains), 0)
    self.assertEqual(len(splits), 0)

  def testLastOneEmpty(self):
    with self.cached_session() as sess:
      # The data looks like the following:
      # Example |  Gradients    | Partition | Feature ID     |
      # i0      |  (0.2, 0.12)  | 0         | 1,2            |
      # i1      |  (-0.5, 0.07) | 0         |                |
      # i2      |  (1.2, 0.2)   | 0         | 2              |
      # i3      |  (4.0, 0.13)  | 1         |                |
      gradients = array_ops.constant([0.2, -0.5, 1.2, 4.0])
      hessians = array_ops.constant([0.12, 0.07, 0.2, 0.13])
      partition_ids = [0, 0, 0, 1]
      indices = [[0, 0], [0, 1], [2, 0]]
      values = array_ops.constant([1, 2, 2], dtype=dtypes.int64)

      gradient_shape = tensor_shape.scalar()
      hessian_shape = tensor_shape.scalar()
      class_id = -1

      split_handler = categorical_split_handler.EqualitySplitHandler(
          l1_regularization=0.1,
          l2_regularization=1,
          tree_complexity_regularization=0,
          min_node_weight=0,
          sparse_int_column=sparse_tensor.SparseTensor(indices, values, [4, 1]),
          feature_column_group_id=0,
          gradient_shape=gradient_shape,
          hessian_shape=hessian_shape,
          multiclass_strategy=learner_pb2.LearnerConfig.TREE_PER_CLASS,
          init_stamp_token=0)
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
        are_splits_ready, partitions, gains, splits = (
            split_handler.make_splits(0, 1, class_id))
        are_splits_ready, partitions, gains, splits = (
            sess.run([are_splits_ready, partitions, gains, splits]))
    self.assertTrue(are_splits_ready)
    self.assertAllEqual([0], partitions)

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


if __name__ == "__main__":
  googletest.main()
