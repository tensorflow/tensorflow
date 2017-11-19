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
"""Tests for the GTFlow prediction Ops.

The tests cover tree traversal and additive models for single and
multi class problems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.contrib.boosted_trees.python.ops import prediction_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


def _append_to_leaf(leaf, c_id, w):
  """Helper method for building tree leaves.

  Appends weight contributions for the given class index to a leaf node.

  Args:
    leaf: leaf node to append to.
    c_id: class Id for the weight update.
    w: weight contribution value.
  """
  leaf.sparse_vector.index.append(c_id)
  leaf.sparse_vector.value.append(w)


def _append_multi_values_to_leaf(leaf, c_ids, w):
  """Helper method for building tree leaves with sparse vector of values.

  Appends weight contributions for the given class index to a leaf node.

  Args:
    leaf: leaf node to append to.
    c_ids: list of class ids
    w: corresponding weight contributions for the classes in c_ids
  """
  for i in range(len(c_ids)):
    leaf.sparse_vector.index.append(c_ids[i])
    leaf.sparse_vector.value.append(w[i])


def _append_multi_values_to_dense_leaf(leaf, w):
  """Helper method for building tree leaves with dense vector of values.

  Appends weight contributions to a leaf. w is assumed to be for all classes.

  Args:
    leaf: leaf node to append to.
    w: corresponding weight contributions for all classes.
  """
  for x in w:
    leaf.vector.value.append(x)


def _set_float_split(split, feat_col, thresh, l_id, r_id, feature_dim_id=None):
  """Helper method for building tree float splits.

  Sets split feature column, threshold and children.

  Args:
    split: split node to update.
    feat_col: feature column for the split.
    thresh: threshold to split on forming rule x <= thresh.
    l_id: left child Id.
    r_id: right child Id.
    feature_dim_id: dimension of the feature column to be used in the split.
  """
  split.feature_column = feat_col
  split.threshold = thresh
  split.left_id = l_id
  split.right_id = r_id
  if feature_dim_id is not None:
    split.feature_id = feature_dim_id


def _set_categorical_id_split(split, feat_col, feat_id, l_id, r_id):
  """Helper method for building tree categorical id splits.

  Sets split feature column, feature id and children.

  Args:
    split: categorical id split node.
    feat_col: feature column for the split.
    feat_id: feature id forming rule x == id.
    l_id: left child Id.
    r_id: right child Id.
  """
  split.feature_column = feat_col
  split.feature_id = feat_id
  split.left_id = l_id
  split.right_id = r_id


class PredictionOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    """Sets up the prediction tests.

    Create a batch of two examples having one dense float, two sparse float
    single valued, one sparse float multidimensionl and one sparse int features.
    The data looks like the following:
    | Instance | Dense0 | SparseF0 | SparseF1 | SparseI0 | SparseM
    | 0        |  7     |    -3    |          |    9,1   | __, 5.0
    | 1        | -2     |          | 4        |          |  3, ___
    """
    super(PredictionOpsTest, self).setUp()
    self._dense_float_tensor = np.array([[7.0], [-2.0]])
    self._sparse_float_indices1 = np.array([[0, 0]])
    self._sparse_float_values1 = np.array([-3.0])
    self._sparse_float_shape1 = np.array([2, 1])
    self._sparse_float_indices2 = np.array([[1, 0]])
    self._sparse_float_values2 = np.array([4.0])
    self._sparse_float_shape2 = np.array([2, 1])
    # Multi dimensional sparse float
    self._sparse_float_indices_m = np.array([[0, 1], [1, 0]])
    self._sparse_float_values_m = np.array([5.0, 3.0])
    self._sparse_float_shape_m = np.array([2, 2])

    self._sparse_int_indices1 = np.array([[0, 0], [0, 1]])
    self._sparse_int_values1 = np.array([9, 1])
    self._sparse_int_shape1 = np.array([2, 2])
    self._seed = 123

  def _get_predictions(self,
                       tree_ensemble_handle,
                       learner_config,
                       apply_dropout=False,
                       apply_averaging=False,
                       center_bias=False,
                       reduce_dim=False):
    return prediction_ops.gradient_trees_prediction(
        tree_ensemble_handle,
        self._seed, [self._dense_float_tensor],
        [self._sparse_float_indices1, self._sparse_float_indices2],
        [self._sparse_float_values1, self._sparse_float_values2],
        [self._sparse_float_shape1, self._sparse_float_shape2],
        [self._sparse_int_indices1], [self._sparse_int_values1],
        [self._sparse_int_shape1],
        learner_config=learner_config,
        apply_dropout=apply_dropout,
        apply_averaging=apply_averaging,
        center_bias=center_bias,
        reduce_dim=reduce_dim)

  def testEmptyEnsemble(self):
    with self.test_session():
      # Empty tree ensenble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="empty")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)
      self.assertAllEqual([[0], [0]], result.eval())
      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testBiasEnsembleSingleClass(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)

      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="bias")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)
      self.assertAllClose([[-0.4], [-0.4]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testBiasEnsembleMultiClass(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      leaf = tree.nodes.add().leaf
      _append_to_leaf(leaf, 0, -0.4)
      _append_to_leaf(leaf, 1, 0.9)

      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="multiclass")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 3

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)
      self.assertAllClose([[-0.4, 0.9], [-0.4, 0.9]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testFullEnsembleSingleClass(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree1.nodes.add().leaf, 0, -0.4)

      # Depth 3 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 1, 2)
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_left.split, 0, -20.0,
                       3, 4)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 1.2)
      _set_categorical_id_split(tree2.nodes.add().categorical_id_binary_split,
                                0, 9, 5, 6)
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.7)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)

      # The first example will get bias -0.4 from first tree and
      # leaf 4 payload of -0.9 hence -1.3, the second example will
      # get the same bias -0.4 and leaf 3 payload (sparse feature missing)
      # of 1.2 hence 0.8.
      self.assertAllClose([[-1.3], [0.8]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testFullEnsembleWithMultidimensionalSparseSingleClass(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree1.nodes.add().leaf, 0, -0.4)

      # Depth 3 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      # Use feature column 2 (sparse multidimensional), split on first value
      # node 0.
      _set_float_split(
          tree2.nodes.add().sparse_float_binary_split_default_right.split,
          2,
          7.0,
          1,
          2,
          feature_dim_id=0)
      # Leafs split on second dimension of sparse multidimensional feature.
      # Node 1.
      _set_float_split(
          tree2.nodes.add().sparse_float_binary_split_default_left.split,
          2,
          4.5,
          3,
          4,
          feature_dim_id=1)
      # Node 2.
      _set_float_split(
          tree2.nodes.add().sparse_float_binary_split_default_right.split,
          2,
          9,
          5,
          6,
          feature_dim_id=1)

      # Node 3.
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.6)
      # Node 4.
      _append_to_leaf(tree2.nodes.add().leaf, 0, 1.3)

      # Node 5.
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.1)
      # Node 6.
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.8)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2

      result, dropout_info = prediction_ops.gradient_trees_prediction(
          tree_ensemble_handle,
          self._seed, [self._dense_float_tensor], [
              self._sparse_float_indices1, self._sparse_float_indices2,
              self._sparse_float_indices_m
          ], [
              self._sparse_float_values1, self._sparse_float_values2,
              self._sparse_float_values_m
          ], [
              self._sparse_float_shape1, self._sparse_float_shape2,
              self._sparse_float_shape_m
          ], [self._sparse_int_indices1], [self._sparse_int_values1],
          [self._sparse_int_shape1],
          learner_config=learner_config.SerializeToString(),
          apply_dropout=False,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      # The first example will get bias -0.4 from first tree and
      # leaf 5 payload of -0.1 hence -0.5, the second example will
      # get the same bias -0.4 and leaf 3 payload (0.6) hence 0.2
      self.assertAllClose([[-0.5], [0.2]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testExcludeNonFinalTree(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree1.nodes.add().leaf, 0, -0.4)

      # Depth 3 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = False
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 1, 2)
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_left.split, 0, -20.0,
                       3, 4)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 1.2)
      _set_categorical_id_split(tree2.nodes.add().categorical_id_binary_split,
                                0, 9, 5, 6)
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.7)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2
      learner_config.growing_mode = learner_pb2.LearnerConfig.WHOLE_TREE
      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)

      # All the examples should get only the bias since the second tree is
      # non-finalized
      self.assertAllClose([[-0.4], [-0.4]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testIncludeNonFinalTree(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree1.nodes.add().leaf, 0, -0.4)

      # Depth 3 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = False
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 1, 2)
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_left.split, 0, -20.0,
                       3, 4)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 1.2)
      _set_categorical_id_split(tree2.nodes.add().categorical_id_binary_split,
                                0, 9, 5, 6)
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.7)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2
      learner_config.growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER
      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)

      # The first example will get bias -0.4 from first tree and
      # leaf 4 payload of -0.9 hence -1.3, the second example will
      # get the same bias -0.4 and leaf 3 payload (sparse feature missing)
      # of 1.2 hence 0.8. Note that the non-finalized tree is included.
      self.assertAllClose([[-1.3], [0.8]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testMetadataMissing(self):
    # Sometimes we want to do prediction on trees that are not added to ensemble
    # (for example in
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree.
      tree1 = tree_ensemble_config.trees.add()
      _append_to_leaf(tree1.nodes.add().leaf, 0, -0.4)

      # Depth 3 tree.
      tree2 = tree_ensemble_config.trees.add()
      # We are not setting the tree_ensemble_config.tree_metadata in this test.
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 1, 2)
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_left.split, 0, -20.0,
                       3, 4)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 1.2)
      _set_categorical_id_split(tree2.nodes.add().categorical_id_binary_split,
                                0, 9, 5, 6)
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.7)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2
      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)

      # The first example will get bias -0.4 from first tree and
      # leaf 4 payload of -0.9 hence -1.3, the second example will
      # get the same bias -0.4 and leaf 3 payload (sparse feature missing)
      # of 1.2 hence 0.8.
      self.assertAllClose([[-1.3], [0.8]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  # For TREE_PER_CLASS strategy, predictions size is num_classes-1
  def testFullEnsembleMultiClassTreePerClassStrategy(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree only for second class.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree1.nodes.add().leaf, 1, -0.2)

      # Depth 2 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_right.split, 1, 4.0,
                       1, 2)
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 3, 4)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree2.nodes.add().leaf, 1, 1.2)
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="ensemble_multi_class")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 3
      learner_config.multi_class_strategy = (
          learner_pb2.LearnerConfig.TREE_PER_CLASS)

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=True)
      # The first example will get bias class 1 -0.2 from first tree and
      # leaf 2 payload (sparse feature missing) of 0.5 hence [0.5, -0.2],
      # the second example will get the same bias class 1 -0.2 and leaf 3
      # payload of class 1 1.2 hence [0.0, 1.0].
      self.assertAllClose([[0.5, -0.2], [0, 1.0]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  # For tree-per-class multiclass handling strategies, predictions vec
  # will have the size of the number of classes.
  # This test is when leafs have SPARSE weights stored (class id and
  # contribution).
  def testFullEnsembleMultiNotClassTreePerClassStrategySparseVector(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree only for second class.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_to_leaf(tree1.nodes.add().leaf, 1, -0.2)

      # Depth 2 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_right.split, 1, 4.0,
                       1, 2)
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 3, 4)
      _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
      _append_multi_values_to_leaf(tree2.nodes.add().leaf, [1, 2], [1.2, -0.7])
      _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="ensemble_multi_class")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 3
      learner_config.multi_class_strategy = (
          learner_pb2.LearnerConfig.FULL_HESSIAN)

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=False)
      # The first example will get bias class 1 -0.2 from first tree and
      # leaf 2 payload (sparse feature missing) of 0.5 hence [0.5, -0.2],
      # the second example will get the same bias class 1 -0.2 and leaf 3
      # payload of class 1 1.2 and class 2-0.7 hence [0.0, 1.0, -0.7].
      self.assertAllClose([[0.5, -0.2, 0.0], [0, 1.0, -0.7]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  # For all non-tree-per class multiclass handling strategies, predictions vec
  # will have the size of the number of classes.
  # This test is when leafs have DENSE weights stored (weight for each class)
  def testFullEnsembleMultiNotClassTreePerClassStrategyDenseVector(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Bias tree only for second class.
      tree1 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _append_multi_values_to_dense_leaf(tree1.nodes.add().leaf, [0, -0.2, -2])

      # Depth 2 tree.
      tree2 = tree_ensemble_config.trees.add()
      tree_ensemble_config.tree_metadata.add().is_finalized = True
      _set_float_split(tree2.nodes.add()
                       .sparse_float_binary_split_default_right.split, 1, 4.0,
                       1, 2)
      _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 3, 4)
      _append_multi_values_to_dense_leaf(tree2.nodes.add().leaf, [0.5, 0, 0])
      _append_multi_values_to_dense_leaf(tree2.nodes.add().leaf, [0, 1.2, -0.7])
      _append_multi_values_to_dense_leaf(tree2.nodes.add().leaf, [-0.9, 0, 0])

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_weights.append(1.0)

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="ensemble_multi_class")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 3
      learner_config.multi_class_strategy = (
          learner_pb2.LearnerConfig.FULL_HESSIAN)

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          reduce_dim=False)
      # The first example will get bias class 1 -0.2 and -2 for class 2 from
      # first tree and leaf 2 payload (sparse feature missing) of 0.5 hence
      # 0.5, -0.2], the second example will get the same bias and leaf 3 payload
      # of class 1 1.2 and class 2-0.7 hence [0.0, 1.0, -2.7].
      self.assertAllClose([[0.5, -0.2, -2.0], [0, 1.0, -2.7]], result.eval())

      # Empty dropout.
      self.assertAllEqual([[], []], dropout_info.eval())

  def testDropout(self):
    with self.test_session():
      # Empty tree ensenble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 1000 trees with some weights.
      for i in range(0, 999):
        tree = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_metadata.add().is_finalized = True
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble_config.tree_weights.append(i + 1)

      # Prepare learner/dropout config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.dropout.dropout_probability = 0.5
      learner_config.learning_rate_tuner.dropout.learning_rate = 1.0
      learner_config.num_classes = 2

      # Apply dropout.
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")
      resources.initialize_resources(resources.shared_resources()).run()

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      # We expect approx 500 trees were dropped.
      dropout_info = dropout_info.eval()
      self.assertIn(dropout_info[0].size, range(400, 601))
      self.assertEqual(dropout_info[0].size, dropout_info[1].size)

      for i in range(dropout_info[0].size):
        dropped_index = dropout_info[0][i]
        dropped_weight = dropout_info[1][i]
        # We constructed the trees so tree number + 1 is the tree weight, so
        # we can check here the weights for dropped trees.
        self.assertEqual(dropped_index + 1, dropped_weight)

      # Don't apply dropout.
      result_no_dropout, no_dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=False,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      self.assertEqual(result.eval().size, result_no_dropout.eval().size)
      for i in range(result.eval().size):
        self.assertNotEqual(result.eval()[i], result_no_dropout.eval()[i])

      # We expect none of the trees were dropped.
      self.assertAllEqual([[], []], no_dropout_info.eval())

  def testDropoutCenterBiasNoGrowingMeta(self):
    # This is for normal non-batch mode where ensemble does not contain the tree
    # that is being built currently.
    num_trees = 10
    with self.test_session():
      # Empty tree ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 10 trees with some weights.
      for i in range(0, num_trees):
        tree = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_metadata.add().is_finalized = True
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble_config.tree_weights.append(i + 1)

      # Prepare learner/dropout config.
      learner_config = learner_pb2.LearnerConfig()
      # Drop all the trees.
      learner_config.learning_rate_tuner.dropout.dropout_probability = 1.0
      learner_config.learning_rate_tuner.dropout.learning_rate = 1.0
      learner_config.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")
      resources.initialize_resources(resources.shared_resources()).run()

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      result_center, dropout_info_center = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=True,
          reduce_dim=True)

      dropout_info = dropout_info.eval()
      dropout_info_center = dropout_info_center.eval()

      # With centering, the bias tree is not dropped.
      num_dropped = dropout_info[0].size
      self.assertEqual(num_dropped, num_trees)
      num_dropped_center = dropout_info_center[0].size
      self.assertEqual(num_dropped_center, num_trees - 1)

      result = result.eval()
      result_center = result_center.eval()
      for i in range(result.size):
        self.assertNotEqual(result[i], result_center[i])

      # First dropped tree is a bias tree 0.
      self.assertEqual(0, dropout_info[0][0])
      # Last dropped tree is the last tree.
      self.assertEqual(num_trees - 1, dropout_info[0][num_dropped - 1])

      # First dropped tree is a tree 1.
      self.assertEqual(1, dropout_info_center[0][0])
      # Last dropped tree is the last tree.
      self.assertEqual(num_trees - 1, dropout_info_center[0][num_dropped_center
                                                             - 1])

  def testDropoutCenterBiasWithGrowingMeta(self):
    # This is batch mode where ensemble already contains the tree that we are
    # building. This tree should never be dropped.
    num_trees = 10
    with self.test_session():
      # Empty tree ensenble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 10 trees with some weights.
      for i in range(0, num_trees):
        tree = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_metadata.add().is_finalized = True
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble_config.tree_weights.append(i + 1)

      # Add growing metadata to indicate batch mode.
      tree_ensemble_config.growing_metadata.num_trees_attempted = num_trees
      tree_ensemble_config.growing_metadata.num_layers_attempted = num_trees

      # Prepare learner/dropout config.
      learner_config = learner_pb2.LearnerConfig()
      # Drop all the trees.
      learner_config.learning_rate_tuner.dropout.dropout_probability = 1.0
      learner_config.learning_rate_tuner.dropout.learning_rate = 1.0
      learner_config.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")
      resources.initialize_resources(resources.shared_resources()).run()

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      result_center, dropout_info_center = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=True,
          reduce_dim=True)

      dropout_info = dropout_info.eval()
      dropout_info_center = dropout_info_center.eval()

      # Last tree is never dropped, the bias tree can be dropped.
      num_dropped = dropout_info[0].size
      self.assertEqual(num_dropped, num_trees - 1)
      num_dropped_center = dropout_info_center[0].size
      self.assertEqual(num_dropped_center, num_trees - 2)

      result = result.eval()
      result_center = result_center.eval()
      for i in range(result.size):
        self.assertNotEqual(result[i], result_center[i])

      # First dropped tree is a bias tree 0.
      self.assertEqual(0, dropout_info[0][0])
      # Last dropped tree is not the last tree (not tree num_trees-1).
      self.assertNotEqual(num_trees - 1, dropout_info[0][num_dropped - 1])
      # First dropped tree is a tree 1.
      self.assertEqual(1, dropout_info_center[0][0])
      # Last dropped tree is not the last tree in ensemble.
      self.assertNotEqual(num_trees - 1,
                          dropout_info_center[0][num_dropped_center - 1])

  def testDropoutSeed(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 10 trees with some weights.
      for i in range(0, 999):
        tree = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_metadata.add().is_finalized = True
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble_config.tree_weights.append(i + 1)

      # Prepare learner/dropout config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.dropout.dropout_probability = 0.5
      learner_config.learning_rate_tuner.dropout.learning_rate = 1.0
      learner_config.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="empty")
      resources.initialize_resources(resources.shared_resources()).run()

      _, dropout_info_1 = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      _, dropout_info_2 = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      # Different seed.
      _, dropout_info_3 = prediction_ops.gradient_trees_prediction(
          tree_ensemble_handle,
          112314, [self._dense_float_tensor],
          [self._sparse_float_indices1, self._sparse_float_indices2],
          [self._sparse_float_values1, self._sparse_float_values2],
          [self._sparse_float_shape1, self._sparse_float_shape2],
          [self._sparse_int_indices1], [self._sparse_int_values1],
          [self._sparse_int_shape1],
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      # First seed with centering bias.
      _, dropout_info_4 = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=True,
          reduce_dim=True)

      # The same seed returns the same results.
      self.assertAllEqual(dropout_info_1.eval(), dropout_info_2.eval())
      # Different seeds give diff results.
      self.assertNotEqual(dropout_info_3.eval().shape,
                          dropout_info_2.eval().shape)
      # With centering bias and the same seed does not give the same result.
      self.assertNotEqual(dropout_info_4.eval(), dropout_info_1.eval())
      # With centering bias has 1 less tree dropped (bias tree is not dropped).
      self.assertEqual(
          len(dropout_info_4.eval()[0]) + 1, len(dropout_info_1.eval()[0]))

  def testDropOutZeroProb(self):
    with self.test_session():
      # Empty tree ensenble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 1000 trees with some weights.
      for i in range(0, 999):
        tree = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_metadata.add().is_finalized = True
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble_config.tree_weights.append(i + 1)

      # Dropout with 0 probability.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.dropout.dropout_probability = 0.0
      learner_config.learning_rate_tuner.dropout.learning_rate = 1.0
      learner_config.num_classes = 2

      # Apply dropout, but expect nothing dropped.
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")
      resources.initialize_resources(resources.shared_resources()).run()

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=True,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      result_no_dropout, _ = self._get_predictions(
          tree_ensemble_handle,
          learner_config=learner_config.SerializeToString(),
          apply_dropout=False,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)

      self.assertAllEqual([[], []], dropout_info.eval())
      self.assertAllClose(result.eval(), result_no_dropout.eval())

  def testAveragingAllTrees(self):
    with self.test_session():
      # Empty tree ensenble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      adjusted_tree_ensemble_config = (
          tree_config_pb2.DecisionTreeEnsembleConfig())
      # Add 100 trees with some weights.
      # When averaging is applied, the tree weights will essentially change to
      # 1, 98/99, 97/99 etc, so lets create the ensemble with such weights.
      # too
      total_num = 100
      for i in range(0, total_num):
        tree = tree_ensemble_config.trees.add()
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)

        tree_ensemble_config.tree_metadata.add().is_finalized = True
        tree_ensemble_config.tree_weights.append(1.0)
        # This is how the weight will look after averaging
        copy_tree = adjusted_tree_ensemble_config.trees.add()
        _append_to_leaf(copy_tree.nodes.add().leaf, 0, -0.4)

        adjusted_tree_ensemble_config.tree_metadata.add().is_finalized = True
        adjusted_tree_ensemble_config.tree_weights.append(
            1.0 * (total_num - i) / total_num)

      # Prepare learner config WITH AVERAGING.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2
      learner_config.averaging_config.average_last_percent_trees = 1.0

      # No averaging config.
      learner_config_no_averaging = learner_pb2.LearnerConfig()
      learner_config_no_averaging.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")

      # This is how our ensemble will "look" during averaging
      adjusted_tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=adjusted_tree_ensemble_config.SerializeToString(
          ),
          name="adjusted")

      resources.initialize_resources(resources.shared_resources()).run()

      # Do averaging.
      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config.SerializeToString(),
          apply_averaging=True,
          reduce_dim=True)

      pattern_result, pattern_dropout_info = self._get_predictions(
          adjusted_tree_ensemble_handle,
          learner_config_no_averaging.SerializeToString(),
          apply_averaging=False,
          reduce_dim=True)

      self.assertAllEqual(result.eval(), pattern_result.eval())
      self.assertAllEqual(dropout_info.eval(), pattern_dropout_info.eval())

  def testAveragingSomeTrees(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      adjusted_tree_ensemble_config = (
          tree_config_pb2.DecisionTreeEnsembleConfig())
      # Add 1000 trees with some weights.
      total_num = 100
      num_averaged = 25
      j = 0
      for i in range(0, total_num):
        tree = tree_ensemble_config.trees.add()
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)

        tree_ensemble_config.tree_metadata.add().is_finalized = True
        tree_ensemble_config.tree_weights.append(1.0)

        # This is how the weight will look after averaging - we are adjusting
        # the weights of the last 25 trees
        copy_tree = adjusted_tree_ensemble_config.trees.add()
        _append_to_leaf(copy_tree.nodes.add().leaf, 0, -0.4)

        adjusted_tree_ensemble_config.tree_metadata.add().is_finalized = True
        if i >= 75:
          adjusted_tree_ensemble_config.tree_weights.append(
              1.0 * (num_averaged - j) / num_averaged)
          j += 1
        else:
          adjusted_tree_ensemble_config.tree_weights.append(1.0)

      # Prepare learner config WITH AVERAGING.
      learner_config_1 = learner_pb2.LearnerConfig()
      learner_config_1.num_classes = 2
      learner_config_1.averaging_config.average_last_percent_trees = 0.25

      # This is equivalent.
      learner_config_2 = learner_pb2.LearnerConfig()
      learner_config_2.num_classes = 2
      learner_config_2.averaging_config.average_last_n_trees = 25

      # No averaging config.
      learner_config_no_averaging = learner_pb2.LearnerConfig()
      learner_config_no_averaging.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")

      # This is how our ensemble will "look" during averaging
      adjusted_tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=adjusted_tree_ensemble_config.SerializeToString(
          ),
          name="adjusted")

      resources.initialize_resources(resources.shared_resources()).run()

      result_1, dropout_info_1 = self._get_predictions(
          tree_ensemble_handle,
          learner_config_1.SerializeToString(),
          apply_averaging=True,
          reduce_dim=True)

      result_2, dropout_info_2 = self._get_predictions(
          tree_ensemble_handle,
          learner_config_2.SerializeToString(),
          apply_averaging=True,
          reduce_dim=True)

      pattern_result, pattern_dropout_info = self._get_predictions(
          adjusted_tree_ensemble_handle,
          learner_config_no_averaging.SerializeToString(),
          apply_averaging=False,
          reduce_dim=True)

      self.assertAllEqual(result_1.eval(), pattern_result.eval())
      self.assertAllEqual(result_2.eval(), pattern_result.eval())

      self.assertAllEqual(dropout_info_1.eval(), pattern_dropout_info.eval())
      self.assertAllEqual(dropout_info_2.eval(), pattern_dropout_info.eval())

  def testAverageMoreThanNumTreesExist(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      adjusted_tree_ensemble_config = (
          tree_config_pb2.DecisionTreeEnsembleConfig())
      # When we say to average over more trees than possible, it is averaging
      # across all trees.
      total_num = 100
      for i in range(0, total_num):
        tree = tree_ensemble_config.trees.add()
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)

        tree_ensemble_config.tree_metadata.add().is_finalized = True
        tree_ensemble_config.tree_weights.append(1.0)
        # This is how the weight will look after averaging
        copy_tree = adjusted_tree_ensemble_config.trees.add()
        _append_to_leaf(copy_tree.nodes.add().leaf, 0, -0.4)

        adjusted_tree_ensemble_config.tree_metadata.add().is_finalized = True
        adjusted_tree_ensemble_config.tree_weights.append(
            1.0 * (total_num - i) / total_num)

      # Prepare learner config WITH AVERAGING.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2
      # We have only 100 trees but we ask to average over 250.
      learner_config.averaging_config.average_last_n_trees = 250

      # No averaging config.
      learner_config_no_averaging = learner_pb2.LearnerConfig()
      learner_config_no_averaging.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="existing")

      # This is how our ensemble will "look" during averaging
      adjusted_tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=adjusted_tree_ensemble_config.SerializeToString(
          ),
          name="adjusted")

      resources.initialize_resources(resources.shared_resources()).run()

      result, dropout_info = self._get_predictions(
          tree_ensemble_handle,
          learner_config.SerializeToString(),
          apply_averaging=True,
          reduce_dim=True)

      pattern_result, pattern_dropout_info = self._get_predictions(
          adjusted_tree_ensemble_handle,
          learner_config_no_averaging.SerializeToString(),
          apply_averaging=False,
          reduce_dim=True)

      self.assertAllEqual(result.eval(), pattern_result.eval())
      self.assertAllEqual(dropout_info.eval(), pattern_dropout_info.eval())


class PartitionExamplesOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    """Sets up the prediction tests.

    Create a batch of two examples having one dense float, two sparse float and
    one sparse int features.
    The data looks like the following:
    | Instance | Dense0 | SparseF0 | SparseF1 | SparseI0 |
    | 0        |  7     |    -3    |          |    9,1   |
    | 1        | -2     |          | 4        |          |
    """
    super(PartitionExamplesOpsTest, self).setUp()
    self._dense_float_tensor = np.array([[7.0], [-2.0]])
    self._sparse_float_indices1 = np.array([[0, 0]])
    self._sparse_float_values1 = np.array([-3.0])
    self._sparse_float_shape1 = np.array([2, 1])
    self._sparse_float_indices2 = np.array([[1, 0]])
    self._sparse_float_values2 = np.array([4.0])
    self._sparse_float_shape2 = np.array([2, 1])
    self._sparse_int_indices1 = np.array([[0, 0], [0, 1]])
    self._sparse_int_values1 = np.array([9, 1])
    self._sparse_int_shape1 = np.array([2, 2])

  def testEnsembleEmpty(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      result = prediction_ops.gradient_trees_partition_examples(
          tree_ensemble_handle, [self._dense_float_tensor], [
              self._sparse_float_indices1, self._sparse_float_indices2
          ], [self._sparse_float_values1, self._sparse_float_values2],
          [self._sparse_float_shape1,
           self._sparse_float_shape2], [self._sparse_int_indices1],
          [self._sparse_int_values1], [self._sparse_int_shape1])

      self.assertAllEqual([0, 0], result.eval())

  def testTreeNonFinalized(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Depth 3 tree.
      tree1 = tree_ensemble_config.trees.add()
      _set_float_split(tree1.nodes.add().dense_float_binary_split, 0, 9.0, 1, 2)
      _set_float_split(tree1.nodes.add()
                       .sparse_float_binary_split_default_left.split, 0, -20.0,
                       3, 4)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.2)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.3)
      _set_categorical_id_split(tree1.nodes.add().categorical_id_binary_split,
                                0, 9, 5, 6)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.6)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_metadata.add().is_finalized = False

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      result = prediction_ops.gradient_trees_partition_examples(
          tree_ensemble_handle, [self._dense_float_tensor], [
              self._sparse_float_indices1, self._sparse_float_indices2
          ], [self._sparse_float_values1, self._sparse_float_values2],
          [self._sparse_float_shape1,
           self._sparse_float_shape2], [self._sparse_int_indices1],
          [self._sparse_int_values1], [self._sparse_int_shape1])

      self.assertAllEqual([5, 3], result.eval())

  def testTreeFinalized(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Depth 3 tree.
      tree1 = tree_ensemble_config.trees.add()
      _set_float_split(tree1.nodes.add().dense_float_binary_split, 0, 9.0, 1, 2)
      _set_float_split(tree1.nodes.add()
                       .sparse_float_binary_split_default_left.split, 0, -20.0,
                       3, 4)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.2)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.3)
      _set_categorical_id_split(tree1.nodes.add().categorical_id_binary_split,
                                0, 9, 5, 6)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree1.nodes.add().leaf, 0, 0.6)

      tree_ensemble_config.tree_weights.append(1.0)
      tree_ensemble_config.tree_metadata.add().is_finalized = True

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="full_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      result = prediction_ops.gradient_trees_partition_examples(
          tree_ensemble_handle, [self._dense_float_tensor], [
              self._sparse_float_indices1, self._sparse_float_indices2
          ], [self._sparse_float_values1, self._sparse_float_values2],
          [self._sparse_float_shape1,
           self._sparse_float_shape2], [self._sparse_int_indices1],
          [self._sparse_int_values1], [self._sparse_int_shape1])

      self.assertAllEqual([0, 0], result.eval())


if __name__ == "__main__":
  googletest.main()
